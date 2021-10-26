from dataclasses import dataclass, field
from typing import List, Tuple
from omegaconf import II

import torch.nn as nn

from fairseq_signals.dataclass import Dataclass
from fairseq_signals.models import register_model, BaseModel
from fairseq_signals.modules import ConvFeatureExtraction

@dataclass
class ConvNetConfig(Dataclass):
    extractor_mode: str  = field (
        default = "default",
        metadata = {
            "help": "mode for feature extractor. default has a single group norm with d"
            "groups in the first conv block, whereas layer_norm layer has layer norms in "
            "every block (meant to use with normalize = True)"
        }
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for the convolutional network"}
    )
    conv_feature_layers: str = field(
        default="[(256, 2, 2)] * 4",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=True, metadata={"help": "include bias in conv encoder"}
    )

    input_dim: int = field(
        default=12,
        metadata={"help": "input dimension"}
    )    
    output_dim: int = II("dataset.n_labels")

@register_model("convnet", dataclass=ConvNetConfig)
class ConvNetModel(BaseModel):
    def __init__(self, cfg: ConvNetConfig):
        super().__init__()
        self.cfg = cfg

        conv_layers = eval(cfg.conv_feature_layers)
        self.embed = conv_layers[-1][0]

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=conv_layers,
            in_d=cfg.input_dim,
            dropout=cfg.dropout,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias
        )

        self.final_proj = nn.Linear(self.embed, cfg.output_dim)
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions."""
        return state_dict
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def forward(
        self,
        source,
        padding_mask=None,
        mask_indices=None
    ):
        features = self.feature_extractor(source)
        
        x = features.mean(dim=2)
        x = self.final_proj(x)

        result = {
            "x": x
        }

        return result

    def get_logits(self, net_output, **kwargs):
        logits = net_output["x"]
        return logits
    
    def get_targets(self, sample, net_output):
        return sample["label"].float()