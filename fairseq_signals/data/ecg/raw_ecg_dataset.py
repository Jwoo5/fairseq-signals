import logging
import os
import sys
import io

import scipy.io
import random
import numpy as np
import torch
import torch.nn.functional as F

from fairseq_signals.data.ecg import PERTURBATION_CHOICES, MASKING_LEADS_STRATEGY_CHOICES

from .. import BaseDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes

logger = logging.getLogger(__name__)

class RawECGDataset(BaseDataset):
    def __init__(
        self,
        sample_rate,
        perturbation_mode: PERTURBATION_CHOICES = "none",
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_leads=False,
        leads_to_load=None,
        label=False,
        normalize=False,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.perturbation_mode = perturbation_mode
        self.retain_original = True

        if perturbation_mode == "random_leads_masking":
            self.mask_leads_selection = mask_compute_kwargs["mask_leads_selection"]
            self.mask_leads_prob = mask_compute_kwargs["mask_leads_prob"]
            self.mask_leads_condition = mask_compute_kwargs["mask_leads_condition"]
        self.sizes = []
        self.max_sample_size = (
             max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.pad_leads = pad_leads
        self.leads_to_load = list(
            int(lead) for lead in leads_to_load.replace(' ','').split(',')
        ) if leads_to_load is not None else None

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
    
    @property
    def apply_perturb(self):
        return self.perturbation_mode != 'none'

    def perturb(self, feats):
        if self.perturbation_mode == "random_leads_masking":
            perturbed, original = self._mask_random_leads(feats)
        else:
            raise AssertionError(
                f"self.perturbation_mode={self.perturbation_mode}"
            )
        
        return perturbed, original

    def _mask_random_leads(self, feats):
        perturbed_feats = feats.new_zeros(feats.size())
        if self.mask_leads_selection == "random":
            survivors = np.random.uniform(0, 1, size=12) > self.mask_leads_prob
            perturbed_feats[survivors] = feats[survivors]
        elif self.mask_leads_selection == "conditional":
            (n1, n2) = self.mask_leads_condition
            assert (
                (0 <= n1 and n1 <=6)
                and (0 <= n2 and n2 <= 6)
            ), (n1, n2)
            s1 = np.array(
                random.sample(list(np.arange(6)), 6-n1)
            )
            s2 = np.array(
                random.sample(list(np.arange(6)), 6-n2)
            ) + 6
            perturbed_feats[s1] = feats[s1]
            perturbed_feats[s2] = feats[s2]
        else:
            raise AssertionError(
                f"mask_leads_selection={self.mask_leads_selection}"
            )
        
        return perturbed_feats, feats

    def postprocess(self, feats, curr_sample_rate):
        if self.sample_rate > 0 and curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        feats = feats.float()
        if self.leads_to_load:
            feats = feats[self.leads_to_load, :]
            if self.pad_leads:
                padded = torch.zeros((12, feats.size(-1)))
                padded[self.leads_to_load] = feats
                feats = padded

        if self.normalize:
            feats = feats.float()
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        
        if self.apply_perturb:
            feats = self.perturb(feats)

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
        originals = [s["original"] for s in samples] if self.retain_original else None
        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        
        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        collated_originals = collated_sources.clone() if self.retain_original else None
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
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                if self.retain_original:
                    collated_originals[i] = torch.cat(
                        [originals[i], originals[i].new_full((originals[i].shape[0], -diff,), 0.0)], dim=-1
                    )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
                if originals is not None:
                    collated_originals[i] = self.crop_to_max_size(originals[i], target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.label:
            out["label"] = torch.cat([s["label"] for s in samples])

        if self.retain_original:
            out["original"] = collated_originals

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
        perturbation_mode: PERTURBATION_CHOICES="none",
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_leads=False,
        leads_to_load=None,
        label=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            perturbation_mode=perturbation_mode,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            pad_leads=pad_leads,
            leads_to_load=leads_to_load,
            label=label,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
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

        res = {'id': index}

        ecg = scipy.io.loadmat(path)

        curr_sample_rate = ecg['curr_sample_rate']
        feats = torch.from_numpy(ecg['feats'])
        if self.apply_perturb:
            source, original = self.postprocess(feats, curr_sample_rate)
            res["source"] = source
            if self.retain_original:
                res["original"] = original
        else:
            res["source"] = self.postprocess(feats, curr_sample_rate)

        # res["file_id"] = ecg['file_id'][0]
        # res["age"] = torch.from_numpy(ecg['age'][0])
        # res["sex"] = torch.from_numpy(ecg['sex'][0])

        if self.label:
            res["label"] = torch.from_numpy(ecg['label'])

        return res

    def __len__(self):
        return len(self.fnames)