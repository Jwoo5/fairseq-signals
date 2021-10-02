from argparse import Namespace
import contextlib
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import MISSING, II
from typing import Any, Optional

from fairseq_signals import tasks
from fairseq_signals.utils import checkpoint_utils, utils
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.tasks import Task
from fairseq_signals.models import (
    BaseModel,
    register_model
)

@dataclass
class ClocsDcConfig(Dataclass):
    clocs_path: Optional[str] = field(
        default=None, metadata={"help": "path to clocs model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability inside clocs model"}
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"}
    )

    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune clocs for this many updates"}
    )

    in_d: int = field(
        default = 1,
        metadata = {"help": "input dimension"}
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    output_size: int = II("dataset.n_labels")
    # this holds the loaded clocs args
    clocs_args: Any = None

@register_model("clocs_dc", dataclass=ClocsDcConfig)
class ClocsDcModel(BaseModel):
    def __init__(self, cfg: ClocsDcConfig, clocs_encoder: BaseModel):
        super().__init__()
        self.cfg = cfg
        self.clocs_encoder = clocs_encoder
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: ClocsDcConfig, task: Task):
        """Build a new model instance."""
        clocs_encoder = ClocsEncoder(cfg)
        return cls(cfg, clocs_encoder)
    
    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["encoder_out"]
        return logits
    
    def get_targets(self, sample, net_output):
        return sample["label"].float()
    
    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)
    
    def forward(self, **kwargs):
        x = self.clocs_encoder(**kwargs)
        return x

class ClocsEncoder(BaseModel):
    def __init__(self, cfg: ClocsDcConfig):
        super().__init__()
        
        arg_overrides = {
            "dropout": cfg.dropout,
            "in_d": cfg.in_d,
        }

        assert cfg.no_pretrained_weights or cfg.clocs_path, (
            "Cannot load pretrained weights. "
            "Please pass --clocs_path explicitly"
        )

        if cfg.clocs_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.clocs_path, arg_overrides)
            clocs_args = state.get("cfg", None)
            if clocs_args is None:
                clocs_args = convert_namespace_to_omegaconf(state["args"])
            clocs_args.criterion = None
            clocs_args.lr_scheduler = None
            cfg.clocs_args = clocs_args
        else:
            state = None
            clocs_args = cfg.clocs_args
            if isinstance(clocs_args, Namespace):
                cfg.clocs_args = clocs_args = convert_namespace_to_omegaconf(clocs_args)
        
        assert cfg.normalize == clocs_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        clocs_args.task.data = cfg.data
        task = tasks.setup_task(clocs_args.task)
        model = task.build_model(clocs_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)
        
        self.clocs_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        dim = clocs_args.model.encoder_embed_dim
        dim = dim * 12 if cfg.in_d == 1 else dim
        trg_dim = cfg.output_size

        self.in_d = cfg.in_d

        self.proj = None
        if trg_dim is not None:
            self.proj = nn.Linear(dim, trg_dim)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.constant_(self.proj.bias, 0.0)
        
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
    
    def forward(self, source, patient_id=None, segment=None, padding_mask=None, **kwargs):
        if self.in_d == 1:
            bsz, csz, tsz = source.shape
            source = source.view(-1, 1, tsz)

        clocs_args = {
            "source": source,
            "patient_id": patient_id,
            "segment": segment
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.clocs_model(**clocs_args, **kwargs)

            x = res["encoder_out"]

        if self.in_d == 1:
            x = x.view(bsz, csz, -1).view(bsz, -1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)
        
        return {
            "encoder_out": x, # ...
            "padding_mask": padding_mask
        }

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict