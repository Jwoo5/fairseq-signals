import logging
import os
import sys

import math
import random
import numpy as np
import scipy.io
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

from typing import List, Optional, Union
from fairseq_signals.data.ecg.augmentations import PERTURBATION_CHOICES, MASKING_LEADS_STRATEGY_CHOICES

from .raw_ecg_dataset import FileECGDataset
from fairseq_signals.dataclass import ChoiceEnum

logger = logging.getLogger(__name__)

class PerturbECGDataset(FileECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        **kwargs
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            **kwargs
        )
        self.retain_original = False

    def perturb(self, feats):
        first = super().perturb(feats)
        second = super().perturb(feats)

        return first, second

    def postprocess(self, feats, curr_sample_rate):
        assert feats.shape[0] == 12, feats.shape[0]

        if self.sample_rate > 0 and curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats.float(), feats.shape)

        feats = self.perturb(feats)

        assert isinstance(feats, tuple), feats
        feats = tuple(f.float() for f in feats)

        return feats

    def collator(self, samples):
        flattened_samples = [s[i] for s in samples for i in range(len(s))]
        flattened_samples = [s for s in flattened_samples if s["source"] is not None]
        if len(flattened_samples) == 0:
            return {}

        out = super().collator(flattened_samples)
        return out

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        ecg = scipy.io.loadmat(path)

        curr_sample_rate = ecg['curr_sample_rate']
        feats = torch.from_numpy(ecg['feats'])

        sources = self.postprocess(feats, curr_sample_rate)

        return [
            {
                "id": index,
                "source": sources[i],
            } for i in range(len(sources))
        ]

class _3KGECGDataset(PerturbECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        angle=45,
        scale=1.5,
        mask_ratio=0.5,
        **kwargs
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            perturbation_mode=None,
            pad_leads=False,
            leads_to_load=None,
            label=False,
            **kwargs
        )

        self.angle = angle
        self.scale = scale
        self.mask_ratio = mask_ratio
    
    def _get_other_four_leads(self, I, II):
        """calculate other four leads (III, aVR, aVL, aVF) from the first two leads (I, II)"""
        III = -I + II
        aVR = -(I + II) / 2
        aVL = I - II/2
        aVF = -I/2 + II

        return III, aVR, aVL, aVF

    def perturb(self, feats):
        leads_taken = [0,1,6,7,8,9,10,11]
        other_leads = [2,3,4,5]
        feats = feats[leads_taken]

        D_i = np.array(
            [
                [0.156, -0.009, -0.172, -0.073, 0.122, 0.231, 0.239, 0.193],
                [-0.227, 0.886, 0.057, -0.019, -0.106, -0.022, 0.040, 0.048],
                [0.021, 0.102, -0.228, -0.310, -0.245, -0.063, 0.054, 0.108]
            ]
        )
        D = np.linalg.pinv(D_i)

        vcg = D_i @ feats

        if self.angle:
            angles = np.random.uniform(-self.angle, self.angle, size=6)
            R1 = R.from_euler('zyx', angles[:3], degrees=True).as_dcm()
            R2 = R.from_euler('zyx', angles[3:], degrees=True).as_dcm()
        else:
            R1 = np.diag((1,1,1))
            R2 = np.diag((1,1,1))
        
        if self.scale:
            scales = np.random.uniform(1, self.scale, size=6)
            S1 = np.diag(scales[:3])
            S2 = np.diag(scales[3:])
        else:
            S1 = np.diag((1,1,1))
            S2 = np.diag((1,1,1))
        
        res1 = D @ S1 @ R1 @ vcg
        res2 = D @ S2 @ R2 @ vcg

        sample_size = feats.shape[-1]

        ecg1 = np.zeros((12, sample_size))
        ecg2 = np.zeros((12, sample_size))

        ecg1[leads_taken] = res1
        ecg1[other_leads] = self._get_other_four_leads(res1[0], res1[1])

        ecg2[leads_taken] = res2
        ecg2[other_leads] = self._get_other_four_leads(res2[0], res2[1])

        if self.mask_ratio:
            sample_size = feats.shape[-1]
            offset = math.floor(sample_size * self.mask_ratio)

            start_indices = np.random.randint(0, sample_size, size=24)
            end_indices = np.array(
                [
                    s + offset if s + offset <= sample_size else sample_size
                    for s in start_indices
                ]
            )
            leftovers = np.array(
                [
                    s + offset - sample_size if s + offset > sample_size else 0
                    for s in start_indices
                ]
            )

            for i in range(12):
                ecg1[i, start_indices[i]:end_indices[i]] = 0
                ecg1[i, 0:leftovers[i]] = 0
            
                ecg2[i, start_indices[i+12]:end_indices[i+12]] = 0
                ecg2[i, 0:leftovers[i+12]] = 0
        
        ecg1 = torch.from_numpy(ecg1)
        ecg2 = torch.from_numpy(ecg2)
        return (ecg1, ecg2)

    def collator(self, samples):
        out = super().collator(samples)
        if len(out) == 0:
            return {}

        flattened_samples = [s[i] for s in samples for i in range(len(s))]
        flattened_samples = [s for s in flattened_samples if s["source"] is not None]
        out["patient_id"] = torch.IntTensor([s["patient_id"] for s in flattened_samples])

        return out

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, str(self.fnames[index]))

        ecg = scipy.io.loadmat(path)

        feats = ecg["feats"]
        curr_sample_rate = ecg["curr_sample_rate"]

        sources = self.postprocess(feats, curr_sample_rate)

        patient_id = ecg["patient_id"][0,0]

        return [
            {
                "id": index,
                "source": sources[i],
                "patient_id": patient_id
            } for i in range(len(sources))
        ]