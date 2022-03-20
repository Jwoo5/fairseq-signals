from dataclasses import dataclass, field
from typing  import List, Tuple
from omegaconf import II

import torch
import torch.nn as nn

from fairseq_signals.utils import utils
from fairseq_signals.models import register_model
from fairseq_signals.models.conv_transformer import ConvTransformerConfig, ConvTransformerModel
from fairseq_signals.modules import GatherLayer
from fairseq_signals.distributed import utils as dist_utils

@dataclass
class _3KGConfig(ConvTransformerConfig):
    apply_mask: bool = False

    angle: int = field(
        default=45,
        metadata={"help": "angle of random rotation in VCG view"}
    )
    scale: float = field(
        default=1.5,
        metadata={"help": "random scaling factor"}
    )
    mask_ratio: float = field(
        default=0.5,
        metadata={"help": "ratio of random time masking for each lead"}
    )

@register_model("3kg", dataclass=_3KGConfig)
class _3KGModel(ConvTransformerModel):
    def __init__(self, cfg: _3KGConfig):
        super().__init__(cfg)
        self.cfg = cfg

        if not cfg.apply_mask:
            self.mask_emb = None
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions. """
        return state_dict
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["x"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0
        
        if aggregate:
            logits = torch.div(logits.sum(dim=1), (logits != 0).sum(dim=1))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits
    
    def forward(self, source, **kwargs):
        if len(source.shape) < 3:
            source = source.unsqueeze(1)
        
        res = super().forward(source, **kwargs)

        x = res["x"]
        padding_mask = res["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
        
        if dist_utils.get_data_parallel_world_size() > 1:
            x = torch.cat(GatherLayer.apply(x), dim=0)
            padding_mask = None

        return {
            "x": x,
            "padding_mask": padding_mask,
        }