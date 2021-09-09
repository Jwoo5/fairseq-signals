# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf

from fairseq_signals.data import (    
    FileECGDataset,
    PatientECGDataset
)
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.models.clocs import CLOCS_MODE_CHOICES

from . import Task, register_task
from ..utils import utils
from ..logging import metrics

logger = logging.getLogger(__name__)

@dataclass
class InferredW2vConfig:
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward
    # required for TPU
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")

@dataclass
class ECGPretrainingConfig(Dataclass):
    data: str = field(default = MISSING, metadata = {"help": "path to data directory"})
    # label: bool = field(
    #     default = False,
    #     metadata = {"help": "whether loading the label together or not, used for fine-tuning"}
    # )
    patient_dataset: bool = field(
        default = False,
        metadata = {
            "help": "if true, loads patient dataset (used for contrastive learning with patients)."
                    "This could cause inconsistent memory allocation between multi GPUs since "
                    "the number of samples per patient is various."
        }
    )
    sample_rate: int = field(
        default = 500,
        metadata = {
            "help": "target sample rate."
        }
    )
    normalize: bool = field(
        default = False,
        metadata = {"help": "if set, normalizes input to have 0 mean and unit variance"}
    )
    enable_padding: bool = field(
        default = False, metadata = {"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default = None, metadata = {"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default = None, metadata = {"help": "min sample size to skip small examples"}
    )
    num_batch_buckets: int = field(
        default = 0,
        metadata = {"help": "number of buckets"}
    )
    precompute_mask_indices: bool = field(
        default = False,
        metadata = {
            "help": "flag to compute mask indices in data preparation"
        }
    )

    clocs_mode: Optional[str] = field(
        default=None, metadata={"help": "coding mode for clocs model"}
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default = None,
        metadata = {
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)"
        }
    )

    # Legacy keys for loading old version of pre-trained model
    max_segment_size: Optional[int] = None
    min_segment_size: Optional[int] = None
    required_segment_size_multiple: Optional[int] = None
    label: Optional[bool] = None

@register_task("ecg_pretraining", dataclass = ECGPretrainingConfig)
class ECGPretrainingTask(Task):
    cfg: ECGPretrainingConfig

    def __init__(
        self,
        cfg: ECGPretrainingConfig
    ):
        super().__init__(cfg)
        
    @classmethod
    def setup_task(cls, cfg: ECGPretrainingConfig, **kwargs):
        """Setup the task 
        
        Args:
            cfg (ECGPretrainingConfig): configuration of this task
        """

        return cls(cfg)
    
    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices:
            assert(
                cfg.inferred_w2v_config is not None
            ), "inferred_w2v_config must be set"
            return OmegaConf.to_container(
                cfg.inferred_w2v_config, resolve = True, enum_to_str = True
            )
        else:
            return {}
    
    def load_dataset(self, split: str, task_cfg: Dataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        if getattr(task_cfg, "patient_dataset", False):
            self.datasets[split] = PatientECGDataset(
                manifest_path = manifest_path,
                split = split,
                sample_rate = task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size = self.cfg.max_sample_size,
                min_sample_size = self.cfg.min_sample_size,
                clocs_mode=task_cfg.clocs_mode,
                pad = task_cfg.enable_padding,
                label = False,
                normalize = task_cfg.normalize,
                num_buckets = self.cfg.num_batch_buckets,
                compute_mask_indices = self.cfg.precompute_mask_indices,
                **self._get_mask_precompute_kwargs(task_cfg)
            )
        else:
            self.datasets[split] = FileECGDataset(
                manifest_path = manifest_path,
                sample_rate = task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size = self.cfg.max_sample_size,
                min_sample_size = self.cfg.min_sample_size,
                pad = task_cfg.enable_padding,
                label = False,
                normalize = task_cfg.normalize,
                num_buckets = self.cfg.num_batch_buckets,
                compute_mask_indices = self.cfg.precompute_mask_indices,
                **self._get_mask_precompute_kwargs(task_cfg)
            )


    def max_positions(self):
        """Maximum input length supported by the encoder,"""
        return (sys.maxsize, sys.maxsize)
    
    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output
    
    def build_model(self, model_cfg: Dataclass):
        model = super().build_model(model_cfg)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if "w2v_args" in actualized_cfg:
                model_cfg.w2v_args = actualized_cfg.w2v_args
        
        return model