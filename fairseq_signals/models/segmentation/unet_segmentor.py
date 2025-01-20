from dataclasses import dataclass, field
from omegaconf import MISSING, II
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals.models import register_model
from fairseq_signals.models.unet import (
    ECGUNetModel,
    ECGUNet3PlusModel,
    ECGUNetFinetuningConfig,
    ECGUNetFinetuningModel,
)

@dataclass
class ECGUNetSegmentationConfig(ECGUNetFinetuningConfig):
    label_idx: Dict[str, int] = field(
        default_factory=lambda: {"p": 0, "qrs": 1, "t": 2, "none": 3},
        metadata={
            "help": "label indices for each of classes: p, qrs, t, and none. used when getting "
                "the final result of the ECG delineation"
        }
    )
    sample_rate: int = II("task.sample_rate")

@register_model("unet_segmentor", dataclass=ECGUNetSegmentationConfig)
class ECGUNetSegmentationModel(ECGUNetFinetuningModel):
    def __init__(self, cfg: ECGUNetSegmentationConfig, backbone: Union[ECGUNetModel, ECGUNet3PlusModel]):
        super().__init__(cfg, backbone)
        self.label_idx = cfg.label_idx
        self.sample_rate = cfg.sample_rate
        
        self.segment = nn.Conv1d(backbone.final_dim, 4, kernel_size=1, padding=0)

    def get_logits(self, net_output, sample=None, **kwargs):
        logits = net_output["segment_out"]
        if (
            sample is not None and (
                "segment_mask" in sample
                and sample["segment_mask"] is not None
                and sample["segment_mask"].any()
            )
        ):
            logits = logits[~sample["segment_mask"]]
        return logits

    def get_targets(self, sample, net_output, **kwargs):
        targets = sample["segment_label"]
        if (
            sample is not None and (
                "segment_mask" in sample
                and sample["segment_mask"] is not None
                and sample["segment_mask"].any()
            )
        ):
            targets = targets[~sample["segment_mask"]]
        return targets

    def forward(self, source, return_features=False, **kwargs):
        res = super().forward(source=source, return_features=return_features, **kwargs)
        
        x = res["x"]
        x = self.segment(x)

        out = {"segment_out": x.transpose(1, 2)} # (B, 4, T) -> (B, T, 4)
        if return_features:
            out["skip1"] = res["skip1"]
            out["skip2"] = res["skip2"]
            out["skip3"] = res["skip3"]
            out["skip4"] = res["skip4"]
            out["skip5"] = res["skip5"]
        
        return out

    def inference(self, source, **kwargs):
        if source.ndim == 2: # (C, T) -> (1, C, T)
            csz, tsz = source.shape
            source = source.view(1, csz, tsz)
        
        return self.forward(source, **kwargs)

    def delineate_ecg(self, source, **kwargs):
        result = self.inference(source, **kwargs)
        seg_result = result["segment_out"].argmax(dim=-1).cpu().numpy()

        # TODO add label idx
        p_idx = self.label_idx["p"]
        qrs_idx = self.label_idx["qrs"]
        t_idx = self.label_idx["t"]
        none_idx = self.label_idx["none"]

        batch_result = []
        for sample_i in range(len(seg_result)):
            p_onsets = []
            p_offsets = []
            qrs_onsets = []
            qrs_offsets = []
            t_onsets = []
            t_offsets = []
            prev = none_idx
            onset = None
            offset = None
            for i, label in enumerate(seg_result[sample_i]):
                if prev != label:
                    if prev != none_idx:
                        offset = i - 1
                        assert onset is not None
                        if prev == p_idx:
                            p_onsets.append(onset)
                            p_offsets.append(offset)
                        elif prev == qrs_idx:
                            qrs_onsets.append(onset)
                            qrs_offsets.append(offset)
                        elif prev == t_idx:
                            t_onsets.append(onset)
                            t_offsets.append(offset)
                    if label != none_idx:
                        onset = i
                    else:
                        onset = None
                # if it reaches the end
                elif i + 1 == len(seg_result[sample_i]) and onset is not None:
                    offset = i
                    if prev == p_idx:
                        p_onsets.append(onset)
                        p_offsets.append(offset)
                    elif prev == qrs_idx:
                        qrs_onsets.append(onset)
                        qrs_offsets.append(offset)
                    elif prev == t_idx:
                        t_onsets.append(onset)
                        t_offsets.append(offset)
                prev = label
            sample_result = {
                "p_onsets": p_onsets,
                "p_offsets": p_offsets,
                "qrs_onsets": qrs_onsets,
                "qrs_offsets": qrs_offsets,
                "t_onsets": t_onsets,
                "t_offsets": t_offsets
            }
            batch_result.append(sample_result)
        return batch_result

@dataclass
class ECGUNetSegmentationWithCGMConfig(ECGUNetSegmentationConfig):
    num_labels: int = field(
        default=MISSING,
        metadata={
            "help": "number of classes for CGM (Classification-Guided Module)"
        }
    )
    cgm_dim: int = field(
        default=512,
        metadata={
            "help": "output dimension for conv layers in classification-guided module (cgm)"
        }
    )
    cgm_kernel_size: int = field(
        default=17,
        metadata={
            "help": "kernel size for conv layers in classification-guided module (cgm)"
        }
    )
    cgm_dropout: float = field(
        default=0.2,
        metadata={
            "help": "dropout probability for classification-guided module (cgm)"
        }
    )

@register_model("unet_cgm_segmentor", dataclass=ECGUNetSegmentationWithCGMConfig)
class ECGUNetSegmentationWithCGMModel(ECGUNetSegmentationModel):
    """ECG UNet segmentation model with CGM (see https://arxiv.org/pdf/2304.06237)"""

    def __init__(self, cfg: ECGUNetSegmentationWithCGMConfig, backbone: Union[ECGUNetModel, ECGUNet3PlusModel]):
        super().__init__(cfg, backbone)
        assert cfg.cgm_kernel_size % 2 == 1, (
            "`cgm_kernel_size` should be a odd number"
        )

        padding = int((cfg.cgm_kernel_size - 1) / 2)
        self.classify = nn.Sequential(
            nn.BatchNorm1d(sum(backbone.all_dims)),
            nn.GELU(),
            nn.Conv1d(sum(backbone.all_dims), cfg.cgm_dim, cfg.cgm_kernel_size, 1, padding),
            nn.BatchNorm1d(cfg.cgm_dim),
            nn.GELU(),
            nn.Dropout1d(cfg.cgm_dropout),
            nn.Conv1d(cfg.cgm_dim, cfg.cgm_dim, cfg.cgm_kernel_size, 1, padding),
            nn.BatchNorm1d(cfg.cgm_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1), # make (B, 512, 1)
            nn.Flatten(start_dim=1), # make (B, 512)
            nn.Linear(cfg.cgm_dim, cfg.num_labels)
        )
    
    def get_logits(self, net_output, sample=None, **kwargs):
        seg_logits = net_output["segment_out"]
        if (
            sample is not None and (
                "segment_mask" in sample
                and sample["segment_mask"] is not None
                and sample["segment_mask"].any()
            )
        ):
            seg_logits = seg_logits[~sample["segment_mask"]]
        cls_logits = net_output["cls_out"]
        return [seg_logits, cls_logits]

    def get_targets(self, sample, net_output, **kwargs):
        seg_targets = sample["segment_label"]
        if (
            sample is not None and (
                "segment_mask" in sample
                and sample["segment_mask"] is not None
                and sample["segment_mask"].any()
            )
        ):
            seg_targets = seg_targets[~sample["segment_mask"]]
        cls_targets = sample["label"]
        return [seg_targets, cls_targets]
    
    def forward(self, source, **kwargs):
        res = super().forward(source=source, return_features=True, **kwargs)
        
        cls_feat = torch.cat([
            F.avg_pool1d(res["skip1"], 16),
            F.avg_pool1d(res["skip2"], 8),
            F.avg_pool1d(res["skip3"], 4),
            F.avg_pool1d(res["skip4"], 2),
            res["skip5"]
        ], dim=1)
        cls_out = self.classify(cls_feat)
        
        return {
            "segment_out": res["segment_out"],
            "cls_out": cls_out
        }
    
    def inference(self, source, **kwargs):
        if source.ndim == 2: # (C, T) -> (1, C, T)
            csz, tsz = source.shape
            source = source.view(1, csz, tsz)
        
        return self.forward(source, **kwargs)