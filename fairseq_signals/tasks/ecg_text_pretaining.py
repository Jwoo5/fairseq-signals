from dataclasses import dataclass
import logging
import os
import sys

from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, List, Union
from omegaconf import MISSING, II, OmegaConf

from fairseq_signals.data import (
    FileECGTextDataset
)
from fairseq_signals.dataclass import Dataclass

from . import Task, register_task

@dataclass
class ECGTextPretrainingConfig(Dataclass):
    data: str = field(default = MISSING, metadata = {"help": "path to data directory"})

    # ecgs
    sample_rate: Optional[int] = field(
        default = None,
        metadata = {
            "help": "target sample rate. it not set, allow any sample rate."
        }
    )
    filter: bool = field(
        default=False,
        metadata={"help": "if set, apply signal filtering to input signals"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"}
    )
    mean_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to .txt file describing means for each lead channel"}
    )
    std_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to .txt file describing stds for each lead channel"}
    )
    enable_padding: bool = field(
        default=False, metadata = {"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata = {"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata = {"help": "min sample size to skip small examples"}
    )
    
    # texts
    tokenizer: str = field(
        default="bert-base-uncased",
        metadata={"help": "tokenizer name to be loaded from huggingface hub"}
    )
    pad_token: int = field(
        default=0,
        metadata={
            'help': 'token id for the special token [PAD]'
        }
    )
    sep_token: int = field(
        default=102,
        metadata={
            'help': 'token id for the special token [SEP]'
        }
    )
    max_text_size: Optional[int] = field(
        default=512,
        metadata={'help': 'max length of input text to crop to for batching'}
    )
    compute_mlm_indices: bool = field(
        default=False,
        metadata={
            "help": "compute mlm indices during constructing a mini-batch"
        }
    )
    mlm_prob: float = field(
        default=0.15,
        metadata={
            "help": "mlm probability"
        }
    )
    #XXX to be removed
    medvill: bool = field(
        default=False,
    )

    model_name: Optional[str] = II("model._name")

@register_task('ecg_text_pretraining', dataclass=ECGTextPretrainingConfig)
class ECGTextPretrainingTask(Task):
    cfg: ECGTextPretrainingConfig
    
    def __init__(self, cfg: ECGTextPretrainingConfig):
        super().__init__(cfg)
    
    @classmethod
    def setup_task(cls, cfg: ECGTextPretrainingConfig, **kwargs):
        """
        Set up the task
        
        Args:
            cfg (ECGTextPretrainingConfig): configuration of this task
        """
        
        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: Dataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = FileECGTextDataset(
            manifest_path,
            pad=task_cfg.enable_padding,
            tokenizer=task_cfg.tokenizer,
            compute_mlm_indices=task_cfg.compute_mlm_indices,
            mlm_prob=task_cfg.mlm_prob,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            max_text_size=self.cfg.max_text_size,
            filter=task_cfg.filter,
            normalize=task_cfg.normalize,
            mean_path=task_cfg.get("mean_path", self.cfg.mean_path),
            std_path=task_cfg.get("std_path", self.cfg.std_path),
            training=True if "train" in split else False,
            #XXX to be removed
            medvill=task_cfg.medvill
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
        
        return model