import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.resnet import (
    Nejedly2021ResnetModel,
    Nejedly2021ResnetFinetuningConfig,
    Nejedly2021ResnetFinetuningModel
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

@register_model("nejedly2021resnet_classifier", dataclass=Nejedly2021ResnetClassificationConfig)
class Nejedly2021ResnetClassificationModel(Nejedly2021ResnetFinetuningModel):
    def __init__(self, cfg: Nejedly2021ResnetClassificationConfig, encoder: Nejedly2021ResnetModel):
        super().__init__(cfg, encoder)

        self.proj = nn.Linear(cfg.dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
    
    def get_logits(self, net_output, normalize=False):
        logits = net_output["out"]

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits
    
    def get_targets(self, sample, net_output):
        return sample["label"].float()
    
    def forward(self, **kwargs):
        res = super().forward(**kwargs)

        x = res["x"]
        x = self.final_dropout(x)
        x = torch.div(x.sum(dim=2), (x != 0).sum(dim=2))
        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "out": x
        }