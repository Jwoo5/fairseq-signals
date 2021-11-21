import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from fairseq_signals.modules import TransposedConvFeatureExtraction
from fairseq_signals.models import register_model
from fairseq_signals.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from fairseq_signals.models.conv_transformer import (
    ConvTransformerModel,
    ConvTransformerConfig
)

from fairseq_signals.utils import utils

@dataclass
class ConvTransformerReconstructionConfig(ConvTransformerConfig):
    apply_mask: bool = False
    pass

@register_model("conv_transformer_recon", dataclass=ConvTransformerReconstructionConfig)
class ConvTransformerReconstructionModel(ConvTransformerModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
        if not cfg.apply_mask or cfg.mask_prob <= 0:
            self.mask_emb = None

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
        """Upgrade a (possibly old) state dict for new versions."""
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
    
    def forward(self, source, features_only=False, **path_through_kwargs):
        res = super().forward(source, **path_through_kwargs)

        x = res["x"]
        if features_only:
            return {"x": x, "padding_mask": res["padding_mask"], "features": res["features"]}

        x = x.transpose(1,2)
        x = self.transposed_conv_layers(x)

        return {
            "x": x
        }
    
    def extract_features(self, source, **kwargs):
        res = self.forward(source, features_only=True, **kwargs)
        return res
    
    def remove_pretraining_modules(self):
        self.transposed_conv_layers = None

@dataclass
class ConvTransformerReconWithWav2Vec2Config(Wav2Vec2Config, ConvTransformerReconstructionConfig):
    pass

@register_model("conv_transformer_recon_with_wav2vec2", dataclass=ConvTransformerReconWithWav2Vec2Config)
class ConvTransformerReconWithWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, cfg: ConvTransformerReconWithWav2Vec2Config):
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
        """Upgrade a (possibly old) state dict for new versions"""
        return state_dict

    def get_logits(self, net_output):
        return (
            self._get_w2v_logits(net_output),
            self._get_recon_logits(net_output)
        )
    
    def get_targets(self, sample, net_output, expand_steps=True):
        return (
            self._get_w2v_targets(sample, net_output, expand_steps),
            self._get_recon_targets(sample, net_output)
        )

    def _get_recon_logits(self, net_output):
        return net_output["recon_out"]["x"].float()

    def _get_w2v_logits(self, net_output):
        return super().get_logits(net_output["w2v_out"])

    def _get_recon_targets(self, sample, net_output):
        recons = net_output["recon_out"]["x"]
        targets = sample["original"]

        assert targets.dim() == 3, targets.shape

        if targets.size(-1) > recons.size(-1):
            offset = targets.size(-1) - recons.size(-1)
            targets = targets[:, :, :-offset]
        
        return targets.float()

    def _get_w2v_targets(self, sample, net_output, expand_steps=True):
        return super().get_targets(sample, net_output, expand_steps)
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def forward(self, **path_through_kwargs):
        w2v_out = super().forward(**path_through_kwargs)

        x = w2v_out["x"].clone()

        x = x.transpose(1,2)
        x = self.transposed_conv_layers(x)

        recon_out = {"x": x}

        return {
            "w2v_out": w2v_out,
            "recon_out": recon_out,
        }
    
    def remove_pretraining_modules(self):
        super().remove_pretraining_modules()
        self.transposed_conv_layers = None