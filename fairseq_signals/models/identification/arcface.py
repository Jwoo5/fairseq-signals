import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Any, Optional
from omegaconf import II

from fairseq_signals.models import register_model
from fairseq_signals.models.conv_transformer import (
    ConvTransformerFinetuningModel,
    ConvTransformerFinetuningConfig
)
from fairseq_signals.utils import utils


@dataclass
class ArcFaceConfig(ConvTransformerFinetuningConfig):
    # legacy keys for loading old-version model
    # w2v_path: Optional[str] = None
    # clocs_path: Optional[str] = None
    # layerdrop: Optional[float] = None
    # mask_channel_before: Optional[bool] = None
    # w2v_args: Any = None
    # clocs_args: Any = None
    pass

#TODO need to add encoder options? (conv_transformer, conv_rnn, convnet, ...)
@register_model("arcface", dataclass=ArcFaceConfig)
class ArcFaceModel(ConvTransformerFinetuningModel):
    def __init__(self, cfg, encoder):
        super().__init__(cfg, encoder)

        self.kernel = nn.Parameter(
            torch.Tensor(
                cfg.output_size,
                cfg.encoder_embed_dim
            )
        )
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["encoder_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        if aggregate:
            pass
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits

    def get_targets(self, sample, net_output):
        return sample["label"].long()
    
    def forward(self, **kwargs):
        res = super().forward(**kwargs)

        x = res["x"]
        padding_mask = res["padding_mask"]

        x = self.final_dropout(x)
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        # if self.cfg.in_d == 1: # TODO: in case of padded lead
        #     concat
        #     ...

        norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm)

        return {
            "encoder_out": x,
            "padding_mask": padding_mask
        }
    
    def get_cosine_similarity(self, logits):
        norm = torch.norm(self.kernel, dim=1, keepdim=True)
        weights = torch.div(self.kernel, norm)

        return torch.mm(logits, weights.T).clamp(-1,1)