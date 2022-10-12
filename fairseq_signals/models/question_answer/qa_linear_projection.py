import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.qa_transformer import (
    QATransformerModel,
    QATransformerFinetuningConfig,
    QATransformerFinetuningModel
)

from fairseq_signals.utils import utils

@dataclass
class QALinearProjectionConfig(QATransformerFinetuningConfig):
    num_labels: int = field(
        default=MISSING,
        metadata= {
            'help': 'final output size of the linear projection.'
        }
    )

@register_model("qa_linear_projection", dataclass=QALinearProjectionConfig)
class QALinearProjectionModel(QATransformerFinetuningModel):
    def __init__(self, cfg: QALinearProjectionConfig, encoder: QATransformerModel):
        super().__init__(cfg, encoder)

        self.proj = nn.Linear(cfg.encoder_embed_dim * 2, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["out"]
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits

    def get_targets(self, sample, net_output):
        return sample["answer"].float()
    
    def forward(self, ecg, question, **kwargs):
        res = super().forward(ecg=ecg, question=question, **kwargs)

        x = res["x"]
        padding_mask = res["padding_mask"]
        
        x = self.final_dropout(x)

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0

        ecg_feats = x[:, :-question.size(-1)]
        ecg_feats = torch.div(ecg_feats.sum(dim=1), (ecg_feats != 0).sum(dim=1))

        question_feats = x[:, -question.size(-1):]
        question_feats = torch.div(question_feats.sum(dim=1), (question_feats != 0).sum(dim=1))

        x = torch.cat([ecg_feats, question_feats], dim=1)
        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": x,
        }