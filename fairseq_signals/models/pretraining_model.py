from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals.models import BaseModel
from fairseq_signals.tasks import Task
from fairseq_signals.dataclass import Dataclass

logger = logging.getLogger(__name__)

@dataclass
class PretrainingConfig(Dataclass):
    all_gather: bool = field(
        default=False, metadata={"help": "whether or not to apply all gather across different gpus"}
    )

    normalize: bool = II("task.normalize")
    filter: bool = II("task.filter")
    data: str = II("task.data")

    # this holds the loaded pre-trained model args
    args: Any = None

class PretrainingModel(BaseModel):
    def __init__(self, cfg: PretrainingConfig):
        super().__init__()
        self.cfg = cfg

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrate a (possibly old) state dict for new versions."""
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: PretrainingConfig, task: Task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    @classmethod
    def from_pretrained(cls, **kwargs):
        """
        Load a :class:`~fairseq_signals.models.PretrainingModel` from a pre-trained model
        checkpoint.
        """
        raise NotImplementedError("PretrainingModel must implement the from_pretrained method")

    def extract_features(self, **kwargs):
        raise NotImplementedError()

    def get_logits(self, **kwargs):
        raise NotImplementedError()
    
    def get_targets(self, **kwargs):
        raise NotImplementedError()
    
    def forward(self, **kwargs):
        raise NotImplementedError()