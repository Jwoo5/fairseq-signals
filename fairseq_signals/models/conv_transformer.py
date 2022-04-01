
import contextlib
from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals import tasks
from fairseq_signals.utils import checkpoint_utils, utils
from fairseq_signals.data.data_utils import compute_mask_indices
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models import BaseModel, register_model
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task
from fairseq_signals.modules import (
    GradMultiply,
    LayerNorm,
    TransformerEncoder,
    ConvFeatureExtraction
)
from fairseq_signals.dataclass import ChoiceEnum, Dataclass

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])

logger = logging.getLogger(__name__)

@dataclass
class ConvTransformerConfig(Dataclass):
    # convnets
    extractor_mode: str  = field (
        default = "default",
        metadata = {
            "help": "mode for conv feature extractor. default has a single group norm with d"
            "groups in the first conv block, whereas layer_norm layer has layer norms in "
            "every block (meant to use with normalize = True)"
        }
    )
    conv_feature_layers: str = field(
        default="[(256, 2, 2)] * 4",
        metadata={
            "help": "string describing convolutional feature extraction layers "
            "in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    in_d: int = field(
        default=12,
        metadata={"help": "input dimension"}
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )


    # transformers
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
    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
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

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
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

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embeddings"},
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded pre-trained model args
    args: Any = None

@register_model(name="conv_transformer", dataclass=ConvTransformerConfig)
class ConvTransformerModel(BaseModel):
    def __init__(self, cfg: ConvTransformerConfig):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=feature_enc_layers,
            in_d=cfg.in_d,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

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

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.num_updates = 0

        self.final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrate a (possibly old) state dict for new versions."""
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices = None,
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
                    min_masks = 2,
                    no_overlap = self.no_mask_overlap,
                    min_space = self.mask_min_space,
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
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)
        
        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])
        
        return input_lengths.to(torch.long)
    
    def forward(
        self,
        source,
        padding_mask=None,
        mask=False,
        mask_indices=None,
        **kwargs
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        
        features = features.transpose(1,2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:,0]
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype = features.dtype, device = features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device = padding_mask.device),
                    output_lengths - 1
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices
            )
        else:
            x = features
            mask_indices = None
        
        x = self.encoder(x, padding_mask=padding_mask)

        return {"x": x, "padding_mask": padding_mask, "features": unmasked_features}

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask=padding_mask, mask=mask)
        return res

    def get_logits(self, net_output):
        raise NotImplementedError()

    def get_targets(self, net_output):
        raise NotImplementedError()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: ConvTransformerConfig,
        **kwargs,
    ):
        """
        Load a :class:`~fairseq_signals.models.ConvTransformerModel` from a pre-trained model
        checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (ConvTransformerConfig): cfg to override some arguments of pre-trained model
        """
        
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
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.encoder_layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

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

        args.task.data = cfg.data
        task = tasks.setup_task(args.task)
        model = task.build_model(args.model)

        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class ConvTransformerFinetuningConfig(FinetuningConfig, ConvTransformerConfig):
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )

    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )

    # overriding arguments
    dropout: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    mask_length: int = 10
    mask_prob: float = 0.5
    mask_selection: MASKING_DISTRIBUTION_CHOICES = "static"
    mask_other: float = 0
    no_mask_overlap: bool = False
    mask_channel_length: int = 10
    mask_channel_prob: float = 0.0
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = "static"
    mask_channel_other: float = 0
    no_mask_channel_overlap: bool = False
    encoder_layerdrop: float = 0.0
    feature_grad_mult: float = 0.0

    output_size: int = II("task.num_labels")

class ConvTransformerFinetuningModel(FinetuningModel):
    def __init__(self, cfg: ConvTransformerFinetuningConfig, encoder: ConvTransformerModel):
        super().__init__(cfg, encoder)

        if not cfg.apply_mask:
            if hasattr(self.encoder, "mask_emb"):
                self.encoder.mask_emb = None
        self.apply_mask = cfg.apply_mask
        
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: ConvTransformerFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = ConvTransformerModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = ConvTransformerModel(cfg)
        
        return cls(cfg, encoder)
    
    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["encoder_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        if aggregate:
            logits = torch.div(logits.sum(dim=1), (logits != 0).sum(dim=1))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits
    
    def get_targets(self, sample, net_output):
        raise NotImplementedError()
    
    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)
    
    def forward(self, source, padding_mask=None, **kwargs):
        args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)

        return res