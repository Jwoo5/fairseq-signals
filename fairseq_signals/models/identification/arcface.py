import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.ecg_transformer import (
    ECGTransformerFinetuningModel,
    ECGTransformerFinetuningConfig
)
from fairseq_signals.utils import utils


@dataclass
class ArcFaceConfig(ECGTransformerFinetuningConfig):
    num_labels: int=field(
        default=MISSING, metadata={"help": "number of patients to be classified when training"}
    )

@register_model("arcface", dataclass=ArcFaceConfig)
class ArcFaceModel(ECGTransformerFinetuningModel):
    def __init__(self, cfg: ArcFaceConfig, encoder):
        super().__init__(cfg, encoder)

        self.kernel = nn.Parameter(
            torch.Tensor(
                cfg.num_labels,
                cfg.encoder_embed_dim
            )
        )
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["out"]
        
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

        norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm)

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": x
        }
    
    def get_cosine_similarity(self, logits):
        norm = torch.norm(self.kernel, dim=1, keepdim=True)
        weights = torch.div(self.kernel, norm)

        return torch.mm(logits, weights.T).clamp(-1,1)