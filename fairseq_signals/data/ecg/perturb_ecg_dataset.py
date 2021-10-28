import logging
import os
import sys

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

from .raw_ecg_dataset import FileECGDataset

logger = logging.getLogger(__name__)

class PerturbECGDataset(FileECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        label=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            label=label,
            normalize=normalize,
            num_buckets=num_buckets,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs
        )

    def perturb(self, feats):
        raise NotImplementedError()

    def postprocess(self, feats, curr_sample_rate):
        assert feats.shape[0] == 12, feats.shape[0]

        if self.sample_rate > 0 and curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        feats = feats.float()

        feats = self.perturb(feats)

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        
        return feats

class _3KGECGDataset(PerturbECGDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        angle=45,
        scale=1.5,
        mask_ratio=0.5,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        leads_to_load=None,
        label=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_rate=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            leads_to_load=leads_to_load,
            label=label,
            normalize=normalize,
            num_buckets=num_buckets,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs
        )

        self.angle = angle
        self.scale = scale
        self.mask_ratio = mask_ratio
    
    def _get_other_four_leads(self, I, II):
        """calculate other four leads (III, aVR, aVL, aVF) with the first two leads (I, II)"""
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
            R1 = R.from_euler('zyx', angles[:3], degrees=True)
            R2 = R.from_euler('zyx', angles[3:], degrees=True)
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

            """
            start_indices : [24,]
            end_indices: [24,]
            leftovers: [24,]

            TODO
            ecg1: [12, 2500]
            -> [each leads, start_indices[each leads]:end_indices[each leads]] = 0
            + [each leads, 0:leftover[each leads]]
            (considering leftover)

            synthesize_vcg.py 참고
            """
            
            ecg1[range(0,12)]