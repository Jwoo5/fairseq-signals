import logging
import os
import sys

import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Optional, Union
from fairseq_signals.data.ecg import augmentations
from fairseq_signals.data.ecg.augmentations import PERTURBATION_CHOICES
from fairseq_signals.dataclass import ChoiceEnum

from .. import BaseDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes

BUCKET_CHOICE = ChoiceEnum(["uniform"])

logger = logging.getLogger(__name__)

class RawECGDataset(BaseDataset):
    def __init__(
        self,
        sample_rate,
        perturbation_mode: Optional[List[PERTURBATION_CHOICES]]=None,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        pad_leads=False,
        leads_to_load=None,
        label=False,
        normalize=False,
        compute_mask_indices=False,
        leads_bucket=None,
        bucket_selection: BUCKET_CHOICE="uniform",
        training=True,
        **kwargs,
    ):
        super().__init__()

        self.training = training

        self.sample_rate = sample_rate
        self.perturbation_mode = perturbation_mode
        self.retain_original = True if perturbation_mode is not None else False

        self.aug_list = []
        if perturbation_mode is not None:
            p = kwargs.pop("p")
            if hasattr(p, "__len__") and len(p) == 1:
                p = list(p) * len(perturbation_mode)
            elif isinstance(p, float):
                p = [p] * len(perturbation_mode)
                
            for aug, prob in zip(perturbation_mode, p):
                self.aug_list.append(
                    augmentations.instantiate_from_name(aug, p=prob, **kwargs)
                )

        self.sizes = []
        self.max_sample_size = (
             max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.pad_leads = pad_leads
        if leads_to_load:
            leads_to_load = eval(leads_to_load)
            self.leads_to_load = list(map(self.get_lead_index, leads_to_load))
        else:
            self.leads_to_load = list(range(12))

        self.leads_bucket = leads_bucket
        if leads_bucket:
            leads_bucket = eval(leads_bucket)
            self.leads_bucket = list(map(self.get_lead_index, leads_bucket))
        self.bucket_selection = bucket_selection

        assert not (leads_bucket and pad_leads), (
            "Bucketizing multiple leads does not work with lead-padding. "
            "Please check that --pad_leads is unset when using bucketized dataset."
        )

        self.label = label
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        if self.compute_mask_indices:
            self.mask_compute_kwargs = kwargs
            self._features_size_map = {}
            self._C = kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    @property
    def apply_perturb(self):
        return self.perturbation_mode != None

    def get_lead_index(self, lead: Union[int, str]) -> int:
        if isinstance(lead, int):
            return lead
        lead = lead.lower()
        order = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        try:
            index = order.index(lead)
        except ValueError:
            raise ValueError(
                "Please make sure that the lead indicator is correct"
            )
        return index

    def perturb(self, feats):
        new_feats = feats.clone()

        for aug in self.aug_list:
            new_feats = aug(new_feats)

        return new_feats

    def postprocess(self, feats, curr_sample_rate):
        if self.sample_rate > 0 and curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        feats = feats.float()
        if self.leads_to_load:
            if self.leads_bucket:
                leads_bucket = set(self.leads_bucket)
                leads_to_load = set(self.leads_to_load)
                if not leads_bucket.issubset(leads_to_load):
                    raise ValueError(
                        "Please make sure that --leads_bucket is a subset of --leads_to_load."
                    )

                leads_to_load = list(leads_to_load - leads_bucket)
                if self.bucket_selection == "uniform":
                    choice = np.random.choice(self.leads_bucket, size=1)
                else:
                    raise Exception("unknown bucket selection " + self.bucket_selection)
                leads_to_load.extend(choice)
            else:
                leads_to_load = self.leads_to_load

            feats = feats[leads_to_load]
            if self.pad_leads:
                padded = torch.zeros((12, feats.size(-1)))
                padded[self.leads_to_load] = feats
                feats = padded

        if self.normalize:
            feats = feats.float()
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)

        if self.training and self.apply_perturb:
            feats = self.perturb(feats)

        return feats

    def crop_to_max_size(self, wav, target_size):
        size = wav.shape[-1]
        diff = size - target_size
        if diff <= 0:
            return wav
        
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[:,start:end]

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
        originals = None
        if self.retain_original:
            originals = [s["original"] for s in samples]
        sizes = [s.size(-1) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros((len(sources), len(sources[0]), target_size))
        collated_originals = collated_sources.clone() if originals else None
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                if originals:
                    collated_originals[i] = originals[i]
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                if originals:
                    collated_originals[i] = torch.cat(
                        [originals[i], originals[i].new_full((originals[i].shape[0], -diff,), 0.0)], dim=-1
                    )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
                if originals:
                    collated_originals[i] = self.crop_to_max_size(originals[i], target_size)

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.label:
            out["label"] = torch.cat([s["label"] for s in samples])

        if originals:
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
        num_buckets=0,
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            **kwargs
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
                if self.min_sample_size is not None and sz < self.min_sample_size:
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
        res["source"] = self.postprocess(feats, curr_sample_rate)
        if self.retain_original:
            res["original"] = feats

        if self.label:
            res["label"] = torch.from_numpy(ecg['label'])

        return res

    def __len__(self):
        return len(self.fnames)