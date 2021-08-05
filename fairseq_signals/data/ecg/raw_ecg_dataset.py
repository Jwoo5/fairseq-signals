import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from .. import BaseDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes

logger = logging.getLogger(__name__)

class RawECGDataset(BaseDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size = None,
        min_sample_size = 0,
        shuffle = True,
        pad = False,
        label = False,
        normalize = False,
        compute_mask_indices = False,
        **mask_compute_kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
             max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.label = label
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])
    
    def __getitem__(self, index):
        raise NotImplementedError()
    
    def __len__(self):
        return len(self.sizes)
    
    def postprocess(self, feats, curr_sample_rate):
        # if feats.dim() == 2:
        #     feats = feats.mean(-1)
        
        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")
        
        # assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats
    
    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav
        
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]
    
    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks = 2,
                no_overlap = self.mask_compute_kwargs["no_mask_overlap"],
                min_space = self.mask_compute_kwargs["mask_min_space"]
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap = self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space = self.mask_compute_kwargs["mask_channel_min_space"]
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )
        
        return mask_indices, mask_channel_indices
    
    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value = value)
    
    def collator(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}
        
        sources = [s["source"] for s in samples]
        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        
        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
        
        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.label:
            out["label"] = torch.cat([s["label"] for s in samples])

        if self.pad:
            input["padding_mask"] = padding_mask
        
        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._buckted_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)
        
        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:,:-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out
    
    def _get_mask_indices_dims(self, size, padding = 0, dilation = 1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1 ) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]
    
    def num_tokens(self, index):
        return self.size(index)
    
    def size(self, index):
        """Return an examples's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            # NOTE: sort according to the size of each sample
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size
                )
            )
            return np.lexsort(order)[::-1]
            # return order[0]
        else:
            return np.arange(len(self))
    
    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the ecg dataset: "
                f"{self.buckets}"
            )

class FileECGDataset(RawECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size = None,
        min_sample_size = 0,
        shuffle = True,
        pad = False,
        label = False,
        normalize = False,
        num_buckets = 0,
        compute_mask_indices = False,
        **mask_compute_kwargs
    ):
        super().__init__(
            sample_rate = sample_rate,
            max_sample_size = max_sample_size,
            min_sample_size = min_sample_size,
            shuffle = shuffle,
            pad = pad,
            label = label,
            normalize = normalize,
            compute_mask_indices = compute_mask_indices,
            **mask_compute_kwargs
        )

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()


        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype = np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass
        
        self.set_bucket_info(num_buckets)
    
    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        # _path, slice_ptr = parse_path(path_or_fp)        
        # if len(slice_ptr) == 2:
        #     byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        #     assert is_sf_audio_data(byte_data)
        #     path_or_fp = io.BytesIO(byte_data)
        
        import scipy.io

        res = {'id': index}

        #TODO handle files not in case of .mat
        ecg = scipy.io.loadmat(path)
        
        #NOTE preprocess data to match with given keys: "feats", "curr_sample_rate", "label"
        feats = torch.from_numpy(ecg['feats'])
        curr_sample_rate = ecg['curr_sample_rate']
        res["source"] = self.postprocess(feats, curr_sample_rate)
        res["patient_id"] = ecg['patient_id'][0]
        res["age"] = torch.from_numpy(ecg['age'][0])
        res["sex"] = torch.from_numpy(ecg['sex'][0])

        if self.label:
            res["label"] = torch.from_numpy(ecg['label'])


        return res

    def __len__(self):
        return len(self.fnames)

class PatientECGDataset(RawECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size = None,
        min_sample_size = 0,
        max_segment_size = None,
        min_segment_size = 0,
        required_segment_size_multiple = 2,
        shuffle = True,
        pad = False,
        label = False,
        normalize = False,
        num_buckets = 0,
        compute_mask_indices = False,
        **mask_compute_kwargs
    ):
        super().__init__(
            sample_rate = sample_rate,
            max_sample_size = max_sample_size,
            min_sample_size = min_sample_size,
            shuffle = shuffle,
            pad = pad,
            label = label,
            normalize = normalize,
            compute_mask_indices = compute_mask_indices,
            **mask_compute_kwargs
        )
        self.max_segment_size = (
            max_segment_size if max_segment_size is not None else sys.maxsize
        )
        self.min_segment_size = min_segment_size

        skipped = 0
        self.fnames = []
        self.num_segments = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            self.ext = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 3, line
                sz = int(items[1])
                seg_sz = int(items[2])
                if (
                    (min_sample_size is not None and sz < min_sample_size)
                    or (min_segment_size is not None and seg_sz < min_segment_size)
                ):
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                seg_sz -= (seg_sz % required_segment_size_multiple)
                self.num_segments.append(seg_sz)
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype = np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarraw array. Please install pyarrow for better performance"
            )
            pass
        
        self.set_bucket_info(num_buckets)
    
    def collator(self, samples):
        collated_samples = [s[i] for s in samples for i in range(len(s))]
        collated_samples = [s for s in collated_samples if s["source"] is not None]
        if len(collated_samples) == 0:
            return {}

        sources = [s["source"] for s in collated_samples]

        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        
        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
        
        input = {"source": collated_sources}
        input["patient_id"] = np.array([s["patient_id"] for s in collated_samples])
        input["segment"] = torch.LongTensor([s["segment"] for s in collated_samples])
        out = {"id": torch.LongTensor([s["id"] for s in collated_samples])}
        if self.label:
            out["label"] = torch.cat([s["label"] for s in collated_samples])

        if self.pad:
            input["padding_mask"] = padding_mask
        
        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._buckted_sizes[s["id"]] for s in collated_samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)
        
        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:,:-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def __getitem__(self, index):
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        paths = [os.path.join(
                self.root_dir,
                str(fn + f"_{i}.{self.ext}")
                ) for i in range(min(self.num_segments[index], self.max_segment_size))]

        feats = []
        labels = []
        patient_ids = []
        ages = []
        sexes = []
        segments = []
        
        import scipy.io

        #TODO handle files not in case of .mat
        for i, path in enumerate(paths):
            ecg = scipy.io.loadmat(path)
        
            #NOTE preprocess data to match with given keys: "feats", "curr_sample_rate", "label"
            feat = torch.from_numpy(ecg['feats'])
            curr_sample_rate = ecg['curr_sample_rate']

            feats.append(
                self.postprocess(feat, curr_sample_rate)
            )
            if self.label:
                labels.append(
                    torch.from_numpy(ecg['label'])
                )
            patient_ids.append(ecg['patient_id'][0])
            ages.append(ecg['age'][0,0])
            sexes.append(ecg['sex'][0,0])
            segments.append(i % 2)

        return [
                {
                    "id": index,
                    "source": feats[i],
                    "label": labels[i] if self.label else None,
                    "patient_id": patient_ids[i],
                    "age": ages[i],
                    "sex": sexes[i],
                    "segment": segments[i]
            } for i in range(len(feats))
        ]
    
    def __len__(self):
        return len(self.fnames)