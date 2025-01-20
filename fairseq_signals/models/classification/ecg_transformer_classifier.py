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
class ECGTransformerClassificationConfig(ECGTransformerFinetuningConfig):
    num_labels: int = field(
        default=MISSING, metadata={"help": "number of labels to be classified"}
    )

@register_model("ecg_transformer_classifier", dataclass=ECGTransformerClassificationConfig)
class ECGTransformerClassificationModel(ECGTransformerFinetuningModel):
    def __init__(self, cfg, encoder):
        super().__init__(cfg, encoder)

        self.proj = nn.Linear(cfg.encoder_embed_dim, cfg.num_labels)
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
        padding_mask = res["padding_mask"]

        x = self.final_dropout(x)
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0

        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": x,
            "saliency": None if res["saliency"] is None else res["saliency"].detach(),
        }