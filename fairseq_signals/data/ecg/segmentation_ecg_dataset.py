import logging
import os

import scipy.io
import torch
import torch.nn.functional as F

from fairseq_signals.data.ecg.raw_ecg_dataset import FileECGDataset


logger = logging.getLogger(__name__)

class SegmentationECGDataset(FileECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        num_buckets=0,
        **kwargs
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            num_buckets=num_buckets,
            **kwargs
        )
    
    def postprocess(self, feats, curr_sample_rate):
        if (
            (self.sample_rate is not None and self.sample_rate > 0)
            and curr_sample_rate != self.sample_rate
        ):
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats.float(), feats.shape)
        
        if self.training:
            feats = self.perturb(feats)
        return feats

    def collator(self, samples):
        out = super().collator(samples)
        if len(out) == 0:
            return {}
        samples = [s for s in samples if s["source"] is not None]

        #TODO consider padding_mask when getting samples of different lengths
        #   torch.stack() will not work for those cases currently
        out["segment_label"] = torch.stack([s["segment_label"] for s in samples])
        if "segment_mask" in samples[0]:
            out["segment_mask"] = torch.stack([s["segment_mask"] for s in samples])

        return out
    
    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        res = {"id": index}

        sample = scipy.io.loadmat(path)

        feats = torch.from_numpy(sample["feats"])

        curr_sample_rate = sample["curr_sample_rate"]
        res["source"] = self.postprocess(feats, curr_sample_rate)
        res["segment_label"] = torch.from_numpy(sample["segment_label"].squeeze())
        if "segment_mask" in sample:
            res["segment_mask"] = torch.from_numpy(sample["segment_mask"].squeeze()).to(bool)
        else:
            res["segment_mask"] = torch.zeros(len(res["segment_label"])).to(bool)

        if self.label:
            if self.label_array is not None:
                res["label"] = torch.from_numpy(self.label_array[sample["idx"].squeeze()])
            else:
                res["label"] = torch.from_numpy(sample["label"].squeeze())
        
        return res