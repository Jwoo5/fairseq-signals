
import contextlib
from dataclasses import dataclass, field
from typing import Any
import logging
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals.utils import utils
from fairseq_signals.models import register_model
from fairseq_signals.models.transformer import (
    TransformerConfig,
    TransformerModel,
    TransformerFinetuningConfig,
    TransformerFinetuningModel
)
from fairseq_signals.tasks import Task
from fairseq_signals.modules import (
    GradMultiply,
    GatherLayer,
    LayerNorm,
    ConvFeatureExtraction,
    ConvPositionalEncoding,
)
from fairseq_signals.distributed import utils as dist_utils

logger = logging.getLogger(__name__)

@dataclass
class ECGTransformerConfig(TransformerConfig):
    # convnets
    extractor_mode: str  = field(
        default = "default",
        metadata = {
            "help": "mode for conv feature extractor. 'default' has a single group norm with d"
            "groups in the first conv block, whereas 'layer_norm' has layer norms in "
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

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embeddings"},
    )

@register_model("ecg_transformer", dataclass=ECGTransformerConfig)
class ECGTransformerModel(TransformerModel):
    def __init__(self, cfg: ECGTransformerConfig):
        super().__init__(cfg)

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

        self.feature_grad_mult = cfg.feature_grad_mult
        self.conv_pos = ConvPositionalEncoding(cfg)
        self.layer_norm = LayerNorm(self.embed)
    
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

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
        **kwargs
    ):
        x, padding_mask = self.get_embeddings(source, padding_mask)
        res = self.get_output(x, padding_mask)
        x = res["x"]

        if self.cfg.all_gather and dist_utils.get_data_parallel_world_size() > 1:
            # we should apply padding mask here if all_gather is true since we cannot assure whether
            # or not all the batches across different gpus have padding mask
            if padding_mask is not None and padding_mask.any():
                x[padding_mask] = 0
                padding_mask = None
            x = torch.cat(GatherLayer.apply(x), dim=0)

        ret = {"x": x, "padding_mask": padding_mask, "saliency": res["saliency"]}

        return ret

    def get_embeddings(self, source, padding_mask):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1,2)
        features = self.layer_norm(features)

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
            padding_mask[torch.where(output_lengths == 0)] = 0
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        features = self.dropout_input(features)

        x = features
        x_conv = self.conv_pos(x, channel_first=False)
        x = x + x_conv

        return x, padding_mask

    def get_output(self, x, padding_mask=None):
        res = self.encoder(x, padding_mask=padding_mask)
        return res

    def extract_features(self, source, padding_mask):
        res = self.forward(source, padding_mask=padding_mask)
        return res

    def get_logits(self, net_output, normalize=False, aggregate=False, **kwargs):
        logits = net_output["x"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0
        
        if aggregate:
            logits = torch.div(logits.sum(dim=1), (logits != 0).sum(dim=1))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=1)
        
        return logits

    def get_targets(self, sample, net_output, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: ECGTransformerConfig,
        **kwargs,
    ):
        """
        Load a :class:`~fairseq_signals.models.ECGTransformerModel` from a pre-trained model
        checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (ECGTransformerConfig): cfg to override some arguments of pre-trained model
        """

        arg_overrides = {
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        model = super().from_pretrained(model_path, cfg, arg_overrides)

        return model

@dataclass
class ECGTransformerFinetuningConfig(TransformerFinetuningConfig, ECGTransformerConfig):
    # overriding arguments
    feature_grad_mult: float = 0.0

class ECGTransformerFinetuningModel(TransformerFinetuningModel):
    def __init__(self, cfg: ECGTransformerFinetuningConfig, encoder: ECGTransformerModel):
        super().__init__(cfg, encoder)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: ECGTransformerFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = ECGTransformerModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = ECGTransformerModel(cfg)
        
        return cls(cfg, encoder)
    
    def get_logits(self, net_output, normalize=False, aggregate=False, **kwargs):
        logits = net_output["encoder_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        if aggregate:
            logits = torch.div(logits.sum(dim=1), (logits != 0).sum(dim=1))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits
    
    def get_targets(self, sample, net_output, **kwargs):
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
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)

        return res