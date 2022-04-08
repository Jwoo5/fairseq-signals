import logging
import os

import scipy.io
import numpy as np
import torch

from typing import List, Optional, Union

from fairseq_signals.data.ecg.raw_ecg_dataset import RawECGDataset

logger = logging.getLogger(__name__)

class IdentificationECGDataset(RawECGDataset):
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
        pids = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 3, line
                sz = int(items[1])
                pid = int(items[2])
                if self.min_sample_size is not None and sz < self.min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(sz)
                pids.append(pid)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)
        self.pids = np.array(pids, dtype=np.int64)

        try:
            import pyarrow
            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass
        
        self.set_bucket_info(num_buckets)
    
    def collator(self, samples):
        out = super().collator(samples)
        if len(out) == 0:
            return {}
        samples = [s for s in samples if s["source"] is not None]
        out["patient_id"] = torch.IntTensor([s["patient_id"] for s in samples])

        return out

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        res = {"id": index}

        ecg = scipy.io.loadmat(path)

        feats = torch.from_numpy(ecg["feats"])
        
        curr_sample_rate = ecg["curr_sample_rate"]
        res["source"] = self.postprocess(feats, curr_sample_rate)
        res["patient_id"] = ecg["patient_id"][0,0]

        if self.label:
            res["label"] = torch.tensor([self.pids[index]])

        return res