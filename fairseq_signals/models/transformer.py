from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals import tasks
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.data.data_utils import compute_mask_indices
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models import BaseModel
from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.modules import (
    TransformerEncoder,
)
from fairseq_signals.tasks import Task
from fairseq_signals.dataclass import ChoiceEnum, Dataclass

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])

logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig(PretrainingConfig):
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a transformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

class TransformerModel(PretrainingModel):
    def __init__(self, cfg: TransformerConfig):
        super().__init__(cfg)

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.num_updates = 0

        self.encoder = TransformerEncoder(cfg)

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap = self.no_mask_channel_overlap,
                min_space = self.mask_channel_min_space
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward(
        self,
        x,
        padding_mask=None,
        **kwargs
    ):
        raise NotImplementedError()

    def extract_features(self, source, padding_mask):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: TransformerConfig,
        arg_appended=None,
        **kwargs,
    ):
        """
        Load a :class:`~fairseq_signals.models.TransformerModel` from a pre-trained model
        checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (TransformerConfig): cfg to override some arguments of pre-trained model
            arg_appended (dict): dict to be appended to cfg
        """

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "encoder_layerdrop": cfg.encoder_layerdrop,
        }
        if arg_appended is not None:
            arg_overrides.update(arg_appended)

        state = checkpoint_utils.load_checkpoint_to_cpu(model_path, arg_overrides)
        args = state.get("cfg", None)
        if args is None:
            args = convert_namespace_to_omegaconf(state["args"])
        args.criterion = None
        args.lr_scheduler = None
        cfg.args = args

        assert cfg.normalize == args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )
        #XXX
        # temporary hack for loading legacy
        if "filter" not in args.task:
            from omegaconf import open_dict
            with open_dict(args.task):
                args.task.filter = False
        #XXX
        assert cfg.filter == args.task.filter, (
            "Fine-tuning works best when signal filtering for data is the same. "
            "Please check that --filter is set or unset for both pre-training and here"
        )

        args.task.data = cfg.data
        task = tasks.setup_task(args.task)
        model = task.build_model(args.model)

        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class TransformerFinetuningConfig(FinetuningConfig, TransformerConfig):
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )

    # overriding arguments
    dropout: float = 0.0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    attention_dropout: float = 0.0
    encoder_layerdrop: float = 0.0

class TransformerFinetuningModel(FinetuningModel):
    def __init__(self, cfg: TransformerFinetuningConfig, encoder: TransformerModel):
        super().__init__(cfg, encoder)

        if hasattr(self.encoder, 'mask_emb'):
            self.encoder.mask_emb = None
        
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: TransformerFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = TransformerModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = TransformerModel(cfg)
        
        return cls(cfg, encoder)