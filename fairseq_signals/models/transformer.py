from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals.data.data_utils import compute_mask_indices
from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.modules import (
    TransformerEncoder,
)
from fairseq_signals.tasks import Task
from fairseq_signals.dataclass import ChoiceEnum

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
    saliency: bool = field(
        default=False, metadata={"help": "whether to extract self-attention weights"}
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

    # masking
    apply_mask: bool = field(default=False, metadata={"help": "whether or not to apply mask"})
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions)"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

class TransformerModel(PretrainingModel):
    def __init__(self, cfg: TransformerConfig):
        super().__init__(cfg)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        if cfg.apply_mask:
            self.mask_emb = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )

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
            "all_gather": False,
        }
        if arg_appended is not None:
            arg_overrides.update(arg_appended)

        if cfg.all_gather:
            logging.warn("model.all_gather overrided to be False when finetuning!")
        
        return super().from_pretrained(model_path, cfg, arg_overrides, **kwargs)

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

        if not cfg.apply_mask and hasattr(self.encoder, "mask_emb"):
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