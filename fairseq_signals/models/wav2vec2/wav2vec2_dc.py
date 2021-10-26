from argparse import Namespace
import contextlib
import logging

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
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
from fairseq_signals.models.conv_transformer import MASKING_DISTRIBUTION_CHOICES

logger = logging.getLogger(__name__)

@dataclass
class Wav2Vec2DcConfig(Dataclass):
    w2v_path: Optional[str] = field(
        default = None, metadata = {"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default = False, metadata = {"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default = 0.0,
        metadata = {
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        }
    )
    activation_dropout: float = field(
        default = 0.0,
        metadata = {
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        }
    )

    # wav2vec2 model
    conv_feature_layers: str = field(
        default="[(256, 2, 2)] * 4",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
            "only override when --no_pretrained_weights is True"
        },
    )
    encoder_layers: int = field(
        default=12,
        metadata={
            "help": "num encoder layers in the transformer inside wav2vec 2.0 model"
                    "only override when --no_pretrained_weights is True"
        }
    )
    encoder_embed_dim: int = field(
        default=768,
        metadata={
            "help": "encoder embedding dimension inside wav2vec 2.0 model"
                    "only override when --no_pretrained_weights is True"
        }
    )
    encoder_ffn_embed_dim: int = field(
        default=3072,
        metadata={
            "help": "encoder embedding dimension for FFN inside wav2vec 2.0 model"
                    "only override when --no_pretrained_weights is True"
        }
    )
    encoder_attention_heads: int = field(
        default=12,
        metadata={
            "help": "num encoder attention heads inside wav2vec 2.0 model"
                    "only override when --no_pretrained_weights is True"
        }
    )

    # masking
    apply_mask: bool = field(
        default = False, metadata = {"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default = 10, metadata = {"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default = 0.5,
        metadata = {"help": "probability of replacing a toekn with mask (normalized by length)"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default = "static", metadata = {"help": "how to choose masks"}
    )
    mask_other: float = field(
        default = 0,
        metadata = {
            "help": "secondary mask argument (used for more complex distributions),"
            "see help in compute_mask_indices"
        }
    )
    no_mask_overlap: bool = field(
        default = False, metadata = {"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default = 10, metadata = {"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default = 0.0, metadata = {"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default = "static",
        metadata = {"help": "how to choose mask length for channel masking"}
    )
    mask_channel_other: float = field(
        default = 0,
        metadata = {
            "help": "secondary mask argument (used for more complex distributions),"
            "see help in compute_mask_indices"
        }
    )
    no_mask_channel_overlap: bool = field(
        default = False, metadata = {"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default = 0, metadata = {"help": "dont finetune wav2vec2 for this many updates"}
    )
    feature_grad_mult: float = field(
        default = 0.0, metadata = {"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default = 0.0, metadata = {"help": "probability of dropping a layer in wav2vec 2.0"}
    )

    in_d: int = field(
        default = 12,
        metadata = {"help": "input dimension"}
    )

    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    output_size: int = II("task.num_labels")
    # this holds the loaded wav2vec args
    w2v_args: Any = None

@register_model("wav2vec2_dc", dataclass = Wav2Vec2DcConfig)
class Wav2Vec2DcModel(BaseModel):
    def __init__(self, cfg: Wav2Vec2DcConfig, w2v_encoder: BaseModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: Wav2Vec2DcConfig, task: Task):
        """Build a new model instance."""
        w2v_encoder = Wav2Vec2Encoder(cfg)
        return cls(cfg, w2v_encoder)
    
    def get_logits(self, net_output, normalize = False, aggregate = False):
        logits = net_output["encoder_out"]

        # TODO need to be checked whether to work properly
        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        # TODO aggregate over tokens to classify the whole outputs
        # example: logits = net_output["encoder_out"].mean(1).float() # B x T x n_classes -> B x n_classes
        #       ... mean is too naive
        if aggregate:
            logits = torch.div(logits.sum(dim = 1), (logits != 0).sum(dim = 1))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim = -1)

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
        x = self.w2v_encoder(**kwargs)
        return x

class Wav2Vec2Encoder(BaseModel):
    def __init__(self, cfg: Wav2Vec2DcConfig):
        super().__init__()
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "in_d": cfg.in_d
        }
        model_overrides = {
            "conv_feature_layers": cfg.conv_feature_layers,
            "encoder_layers": cfg.encoder_layers,
            "encoder_embed_dim": cfg.encoder_embed_dim,
            "encoder_ffn_embed_dim": cfg.encoder_ffn_embed_dim,
            "encoder_attention_heads": cfg.encoder_attention_heads
        }
        assert cfg.no_pretrained_weights or cfg.w2v_path, (
            "Cannot load pretrained weights. "
            "Please pass --w2v_path explicitly."
        )

        #TODO if intended to train from scratch, do not refer model checkpoint anymore.
        #XXX if no_pretrained_weights, init w2v_args on its own.
        if cfg.w2v_args is None:
            if cfg.no_pretrained_weights:
                arg_overrides.update(model_overrides)
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        assert cfg.no_pretrained_weights or cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict = True)
            logger.info(f"Loaded pre-trained model parameters from {cfg.w2v_path}")

        model.remove_pretraining_modules()

        self.w2v_model = model
        if not cfg.apply_mask:
            self.w2v_model.mask_emb = None

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        dim = cfg.w2v_args.model.encoder_embed_dim
        trg_dim = cfg.output_size

        if trg_dim is not None:
            self.proj = nn.Linear(dim, trg_dim)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.constant_(self.proj.bias, 0.0)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask = None, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x, # B x T x n_labels
            "padding_mask": padding_mask,
        }
    
    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict