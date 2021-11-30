from argparse import Namespace
from dataclasses import dataclass, field
from fairseq_signals.models.clocs.clocs import CLOCS_MODE_CHOICES
from omegaconf import II

import torch

from fairseq_signals.dataclass import ChoiceEnum
from fairseq_signals.utils import utils
from fairseq_signals.models import register_model
from fairseq_signals.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
from fairseq_signals.distributed import utils as dist_utils

@dataclass
class Wav2Vec2ClocsConfig(Wav2Vec2Config):
    pass

@register_model("wav2vec2_clocs", dataclass = Wav2Vec2ClocsConfig)
class Wav2Vec2ClocsModel(Wav2Vec2Model):
    def __init__(self, cfg: Wav2Vec2ClocsConfig):
        super().__init__(cfg)
        self.cfg = cfg
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions"""
        return state_dict
    
    def get_features(self, net_output, aggregate=False):
        features = net_output["features"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            features[net_output["padding_mask"]] = 0
        
        if aggregate:
            features = torch.div(features.sum(dim=1), (features != 0).sum(dim=1))

        return features

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def extract_features(self, source, padding_mask, mask=False):
        res = super().forward(source, padding_mask, mask=mask, features_only=True)
        return res
    
    def forward(self, **w2v_kwargs):
        w2v_out = super().forward(return_features=True, **w2v_kwargs)

        return w2v_out