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
from typing import Optional, Any, Tuple, List, Union
from omegaconf import MISSING, II, OmegaConf

from fairseq_signals.data import (    
    FileECGDataset,
    ClocsECGDataset,
    PerturbECGDataset,
    _3KGECGDataset
)
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.data.ecg.raw_ecg_dataset import BUCKET_CHOICE

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
class Inferred3KGConfig:
    # The following are needed to perturb data samples in 3KG model
    angle: Optional[int] = II("model.angle")
    scale: Optional[float] = II("model.scale")
    mask_ratio: Optional[float] = II("model.mask_ratio")

@dataclass
class ECGPretrainingConfig(Dataclass):
    data: str = field(default = MISSING, metadata = {"help": "path to data directory"})
    leads_to_load: Optional[str] = field(
        default=None,
        metadata={
            "help": "string describing list of lead indicators or lead indices to be loaded"
            "note that the sequence of leads is [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]"
            "if not set, load all the available leads of samples"
        }
    )
    leads_bucket: Optional[str] = field(
        default=None,
        metadata={
            "help": "string describing list of lead indicators or lead indices to be bucketized"
            "This set of leads should be a subset of --leads_to_load"
        }
    )
    bucket_selection: BUCKET_CHOICE = field(
        default="uniform",
        metadata={
            "help": "how to bucketize multiple leads"
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
    enable_padding_leads: bool = field(
        default=False, metadata={"help": "pad unavailable leads of samples"}
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

    perturbation_mode: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "mode for perturbation before samples being forwarded. "
            "the perturbation is applied in the order of the list"
        }
    )
    p: List[float] = field(
        default_factory=lambda: [1.0],
        metadata={
            "help": "list of probability of applying each augmentation"
            "if given one element, the probability is applied across all the augmentation"
        }
    )
    max_amplitude: float = field(
        default=0.1,
        metadata={"help": "max amplitude of augmented noises"}
    )
    min_amplitude: float = field(
        default=0,
        metadata={"help": "min amplitude of augmented noises"}
    )
    dependency: bool = field(
        default=True,
        metadata={"help": "whether to apply dependency between frontal leads"}
    )
    shift_ratio: float = field(
        default=0.2,
        metadata={"help": "shifted ratio in baseline shift"}
    )
    num_segment: int = field(
        default=1,
        metadata={"help": "number of segments in baseline shift"}
    )
    max_freq: float = field(
        default=0.2,
        metadata={"help": "max frequency of augmented baseline wandering"}
    )
    min_freq: float = field(
        default=0.01,
        metadata={"help": "min frequency of augmented baseline wandering"}
    )
    k: int = field(
        default=3,
        metadata={"help": "the number of times applying baseline wandering"}
    )
    mask_leads_selection: str = field(
        default="random",
        metadata={
            "help": "how to choose leads to be masked. random is masking every "
            "lead with the probability of --mask_leads_prob. conditional is masking "
            "specific number of leads according to --mask_leads_condition"
        }
    )
    mask_leads_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a lead with 0"
        }
    )
    mask_leads_condition: Tuple[int, int] = field(
        default=(4, 5),
        metadata={
            "help": "specific number of leads to be masked. "
            "tuple of 2 values (# out of the first 6 leads, # out of the last 6 leads)"
        }
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default = None,
        metadata = {
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)"
        }
    )
    inferred_3kg_config: Optional[Inferred3KGConfig] = field(
        default=None,
        metadata={
            "help": "3kg model arguments used to perturb data samples"
        }
    )
    model_name: str = II("model._name")

    clocs_mode: Optional[str] = None

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
    
    def _get_perturbation_kwargs(self):
        return {
            "p": self.cfg.p,
            "max_amplitude": self.cfg.max_amplitude,
            "min_amplitude": self.cfg.min_amplitude,
            "dependency": self.cfg.dependency,
            "shift_ratio": self.cfg.shift_ratio,
            "num_segment": self.cfg.num_segment,
            "max_freq": self.cfg.max_freq,
            "min_freq": self.cfg.min_freq,
            "freq": self.cfg.sample_rate,
            "k": self.cfg.k,
            "mask_leads_selection": self.cfg.mask_leads_selection,
            "mask_leads_prob": self.cfg.mask_leads_prob,
            "mask_leads_condition": self.cfg.mask_leads_condition,
        }

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

        if 'clocs' in task_cfg.model_name:
            self.datasets[split] = ClocsECGDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                perturbation_mode=self.cfg.perturbation_mode,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                clocs_mode=task_cfg.clocs_mode,
                pad=task_cfg.enable_padding,
                pad_leads=task_cfg.enable_padding_leads,
                leads_to_load=task_cfg.leads_to_load,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets,
                compute_mask_indices=self.cfg.precompute_mask_indices,
                leads_bucket=self.cfg.leads_bucket,
                bucket_selection=self.cfg.bucket_selection,
                training=True if 'train' in split else False,
                **self._get_mask_precompute_kwargs(task_cfg),
                **self._get_perturbation_kwargs()
            )
        elif task_cfg.model_name == '3kg':
            if task_cfg.leads_to_load is not None:
                raise AssertionError(
                    "pre-training 3kg must contain all the 12-leads. "
                    "please set --leads_to_load to null"
                )
            inferred_3kg_config = OmegaConf.to_container(
                self.cfg.inferred_3kg_config, resolve=True, enum_to_str=True
            )
            self.datasets[split] = _3KGECGDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets,
                training=True if 'train' in split else False,
                **inferred_3kg_config,
            )
        elif task_cfg.model_name == 'simclr':
            self.datasets[split] = PerturbECGDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                perturbation_mode=self.cfg.perturbation_mode,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                pad_leads=task_cfg.enable_padding_leads,
                leads_to_load=task_cfg.leads_to_load,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets,
                compute_mask_indices=self.cfg.precompute_mask_indices,
                leads_bucket=self.cfg.leads_bucket,
                bucket_selection=self.cfg.bucket_selection,
                training=True if 'train' in split else False,
                **self._get_mask_precompute_kwargs(task_cfg),
                **self._get_perturbation_kwargs()
            )
        else:
            self.datasets[split] = FileECGDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                perturbation_mode=self.cfg.perturbation_mode,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                pad_leads=task_cfg.enable_padding_leads,
                leads_to_load=task_cfg.leads_to_load,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets,
                compute_mask_indices=self.cfg.precompute_mask_indices,
                leads_bucket=self.cfg.leads_bucket,
                bucket_selection=self.cfg.bucket_selection,
                training=True if 'train' in split else False,
                **self._get_mask_precompute_kwargs(task_cfg),
                **self._get_perturbation_kwargs()
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
    
    def valid_step(self, sample, model, criterion, subset=None):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion, subset)
        return loss, sample_size, logging_output
    
    def build_model(self, model_cfg: Dataclass):
        model = super().build_model(model_cfg)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if "w2v_args" in actualized_cfg:
                model_cfg.w2v_args = actualized_cfg.w2v_args
        
        return model