from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals import tasks
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
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
    def from_pretrained(cls, model_path, cfg, arg_overrides=None, **kwargs):
        """
        Load a :class:`~fairseq_signals.models.PretrainingModel` from a pre-trained model
        checkpoint.
        
        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (PretrainingConfig): cfg to override some arguments of pre-trained model
            arg_overrides (dict): a Python dictionary to replace an old arg (key) with a new arg (value)
        """

        state = checkpoint_utils.load_checkpoint_to_cpu(model_path, arg_overrides)
        args = state.get("cfg", None)
        if args is None:
            args = convert_namespace_to_omegaconf(state["args"])
        args.criterion = None
        args.lr_scheduler = None

        assert cfg.normalize == args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )
        assert cfg.filter == args.task.filter, (
            "Fine-tuning works best when signal filtering for data is the same. "
            "Please check that --filter is set or unset for both pre-training and here"
        )

        args.task.data = cfg.data
        task = tasks.setup_task(args.task, from_checkpoint=True)
        model = task.build_model(args.model)

        if hasattr(model, "remove_pretrainined_modules"):
            model.remove_pretraining_modules()
        
        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

    def extract_features(self, **kwargs):
        raise NotImplementedError()

    def get_logits(self, **kwargs):
        raise NotImplementedError()
    
    def get_targets(self, **kwargs):
        raise NotImplementedError()
    
    def forward(self, **kwargs):
        raise NotImplementedError()