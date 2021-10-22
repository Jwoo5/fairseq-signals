import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from fairseq_signals.models import register_model
from fairseq_signals.models.conv_transformer import (
    ConvTransformerFinetuningModel,
    ConvTransformerFinetuningConfig
)

from fairseq_signals.utils import utils

@dataclass
class LinearProjectionConfig(ConvTransformerFinetuningConfig):
    pass

@register_model("linear_projection", dataclass=LinearProjectionConfig)
class LinearProjectionModel(ConvTransformerFinetuningModel):
    def __init__(self, cfg, encoder):
        super().__init__(cfg, encoder)

        self.proj = nn.Linear(cfg.encoder_embed_dim, cfg.output_size)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

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
        return sample["label"].float()
    
    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder(**kwargs)

            x = res["x"]
            padding_mask = res["padding_mask"]
        
        x = self.final_dropout(x)
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        x = self.proj(x)

        return {
            "encoder_out": x,
            "padding_mask": padding_mask
        }