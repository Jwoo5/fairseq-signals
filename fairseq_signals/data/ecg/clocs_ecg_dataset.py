import logging
import os
import sys

import numpy as np
import scipy.io
import torch

from typing import Optional, List, Union
from fairseq_signals.data.ecg.augmentations import PERTURBATION_CHOICES
from .raw_ecg_dataset import RawECGDataset

logger = logging.getLogger(__name__)

class ClocsECGDataset(RawECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        num_buckets=0,
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )
        self.clocs_mode = kwargs['clocs_mode']
        assert self.clocs_mode in ["cmsc", "cmlc", "cmsmlc"]
        self.max_segment_size = sys.maxsize
        self.min_segment_size = 2 if self.clocs_mode in ["cmsc", "cmsmlc"] else 1
        required_segment_size_multiple = 2 if self.clocs_mode in ["cmsc", "cmsmlc"] else 1

        skipped = 0
        self.fnames = []
        self.segments = []
        self.leads = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            self.ext = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 4, line
                sz = int(items[1])
                seg = [int(s) for s in items[3].split(',')][:self.max_segment_size]
                seg_sz = len(seg)
                if (
                    (self.min_sample_size is not None and sz < self.min_sample_size)
                    or (self.min_segment_size is not None and seg_sz < self.min_segment_size)
                ):
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                self.leads.append(int(items[2]))
                self.segments.append(
                    seg if len(seg) % required_segment_size_multiple == 0 else (
                        seg[:-(len(seg) % required_segment_size_multiple)]
                    )
                )
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
        flattened_samples = [s[i] for s in samples for i in range(len(s))]
        flattened_samples = [s for s in flattened_samples if s["source"] is not None]
        if len(flattened_samples) == 0:
            return {}

        sources = [s["source"] for s in flattened_samples]

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
                    [source, source.new_full((source.shape[0], -diff,), 0.0)], dim=-1
                )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)        

        input = {"source": collated_sources}
        out = {"id": torch.LongTensor([s["id"] for s in flattened_samples])}
        out["patient_id"] = torch.IntTensor([s["patient_id"] for s in flattened_samples])
        out["segment"] = torch.IntTensor([s["segment"] for s in flattened_samples])
        if self.label:
            out["label"] = torch.cat([s["label"] for s in flattened_samples])

        if self.pad:
            input["padding_mask"] = padding_mask
        
        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._buckted_sizes[s["id"]] for s in flattened_samples)
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
                ) for i in self.segments[index]
        ]
        res = []
        for i, path in enumerate(paths):
            out = {"id": index}
            ecg = scipy.io.loadmat(path)
        
            feats = torch.from_numpy(ecg['feats'])
            curr_sample_rate = ecg['curr_sample_rate']

            out["source"] = self.postprocess(feats, curr_sample_rate)
            if self.label:
                out["label"] = torch.from_numpy(ecg["label"])
            out["patient_id"] = ecg["patient_id"][0,0]
            out["age"] = ecg["age"][0,0]
            out["sex"] = ecg["sex"][0,0]
            out["segment"] = i % 2

            res.append(out)

        return res
    
    def __len__(self):
        return len(self.fnames)