from dataclasses import dataclass, field
from omegaconf import MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals.models import register_model
from fairseq_signals.models.resnet import (
    Nejedly2021ResnetModel,
    Nejedly2021ResnetFinetuningConfig,
    Nejedly2021ResnetFinetuningModel,
)
from fairseq_signals.models.se_wrn import (
    SEWideResidualNetworkModel,
    SEWideResidualNetworkFinetuningConfig,
    SEWideResidualNetworkFinetuningModel
)

from fairseq_signals.utils import utils

@dataclass
class Nejedly2021ResnetClassificationConfig(Nejedly2021ResnetFinetuningConfig):
    num_labels: int = field(
        default=MISSING,
        metadata={
            "help": "final output size of the linear projection"
        }
    )
    indicate_leads: bool = field(
        default=False,
        metadata={
            "help": "whether to indicate which leads are activated"
        }
    )
    use_mha: bool = field(
        default=False,
        metadata={
            "help": "whether to use the multi-head attention module"
        }
    )

@register_model("nejedly2021resnet_classifier", dataclass=Nejedly2021ResnetClassificationConfig)
class Nejedly2021ResnetClassificationModel(Nejedly2021ResnetFinetuningModel):
    def __init__(self, cfg: Nejedly2021ResnetClassificationConfig, encoder: Nejedly2021ResnetModel):
        super().__init__(cfg, encoder)
        self.indicate_leads = cfg.indicate_leads

        self.mha = None
        if cfg.use_mha:
            self.mha = nn.MultiheadAttention(cfg.num_filters, 8)

        in_dim = cfg.num_filters * 12 if self.cfg.leadwise else cfg.num_filters
        self.proj = nn.Linear(in_dim + 12 if cfg.indicate_leads else in_dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
    
    def get_logits(self, net_output, normalize=False, **kwargs):
        logits = net_output["out"]

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits
    
    def get_targets(self, sample, net_output, **kwargs):
        if isinstance(sample["label"], torch.Tensor):
            return sample["label"].float()
        else:
            return sample["label"]
    
    def forward(self, **kwargs):
        res = super().forward(**kwargs)

        x = res["x"]
        x = self.final_dropout(x)

        bsz = x.shape[0]

        if self.mha is not None and not self.cfg.leadwise:
            x = x.permute(2, 0, 1) # (S, B, dim)
            x, s = self.mha(x, x, x)
            x = x.permute(1, 2, 0) # (B, dim, S)

        x = torch.div(x.sum(dim=-1), (x != 0).sum(dim=-1))

        if self.cfg.leadwise:
            x = x.view(bsz, -1)

        if self.indicate_leads:
            x = torch.cat((x, res["l"]), dim=1)

        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "out": x
        }

@dataclass
class SEWideResidualNetworkClassificationConfig(SEWideResidualNetworkFinetuningConfig):
    num_labels: int = field(
        default=MISSING,
        metadata={
            "help": "final output size of the linear projection"
        }
    )

@register_model("se_wrn_classifier", dataclass=SEWideResidualNetworkClassificationConfig)
class SEWideResidualNetworkClassificationModel(SEWideResidualNetworkFinetuningModel):
    def __init__(self, cfg: SEWideResidualNetworkClassificationConfig, encoder: SEWideResidualNetworkModel):
        super().__init__(cfg, encoder)

        dim = encoder.num_filters[-1] * self.cfg.widening_factor
        self.final_dim = dim * 12 if self.cfg.leadwise else dim

        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.final_dropout)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.proj = nn.Linear(self.final_dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def get_logits(self, net_output, normalize=False, **kwargs):
        logits = net_output["out"]

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits
    
    def get_targets(self, sample, net_output, **kwargs):
        if isinstance(sample["label"], torch.Tensor):
            return sample["label"].float()
        else:
            return sample["label"]

    def forward(self, **kwargs):
        res = super().forward(**kwargs)

        x = res["x"]
        bsz = x.shape[0]
        if self.cfg.leadwise:
            bsz, csz, dim, tsz = x.shape
            x = x.view(bsz * csz, dim, tsz)

        x = self.relu(self.bn(x))
        x = self.dropout(x)

        x = self.maxpool(x).squeeze(-1)
        if self.cfg.leadwise:
            x = x.view(bsz, -1)

        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "out": x
        }