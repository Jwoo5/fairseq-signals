import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from fairseq_signals.modules import TransposedConvFeatureExtraction
from fairseq_signals.models import register_model
from fairseq_signals.models.conv_transformer import (
    ConvTransformerModel,
    ConvTransformerConfig
)

from fairseq_signals.utils import utils

@dataclass
class ConvTransformerReconstructionConfig(ConvTransformerConfig):
    pass

@register_model("conv_transformer_recon", dataclass=ConvTransformerReconstructionConfig)
class ConvTransformerReconstructionModel(ConvTransformerModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        conv_layers = eval(self.cfg.conv_feature_layers)
        conv_layers.reverse()

        transposed_conv_layers = []
        for i, (dim, k, s) in enumerate(conv_layers):
            if i + 1 < len(conv_layers):
                transposed_conv_layers.append((conv_layers[i+1][0],k,s))
            else:
                transposed_conv_layers.append((cfg.in_d, k, s))

        self.transposed_conv_layers = TransposedConvFeatureExtraction(
            conv_transposed_layers=transposed_conv_layers,
            in_d=cfg.encoder_embed_dim,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new vrsions."""
        return state_dict

    def get_logits(self, net_output):
        return net_output["x"].float()

    def get_targets(self, sample, net_output):
        x = net_output["x"]
        target = sample["original"]

        assert target.dim() == 3, target.shape

        if target.size(-1) > x.size(-1):
            offset = target.size(-1) - x.size(-1)
            target = target[:,:,:-offset]

        return target.float()

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def forward(self, **path_through_kwargs):
        res = super().forward(**path_through_kwargs)

        x = res["x"]

        x = x.transpose(1,2)
        x = self.transposed_conv_layers(x)

        return {
            "x": x
        }