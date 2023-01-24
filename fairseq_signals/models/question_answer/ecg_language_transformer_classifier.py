import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.ecg_language_transformer import (
    ECGLanguageTransformerModel,
    ECGLanguageTransformerFinetuningConfig,
    ECGLanguageTransformerFinetuningModel
)

from fairseq_signals.utils import utils

@dataclass
class ECGLanguageTransformerClassificationConfig(ECGLanguageTransformerFinetuningConfig):
    num_labels: int = field(
        default=MISSING,
        metadata= {
            'help': 'final output size of the linear projection'
        }
    )

@register_model("ecg_language_transformer_classifier", dataclass=ECGLanguageTransformerClassificationConfig)
class ECGLanguageTransformerClassificationModel(ECGLanguageTransformerFinetuningModel):
    def __init__(self, cfg: ECGLanguageTransformerClassificationConfig, encoder: ECGLanguageTransformerModel):
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
    
    def forward(self, ecg, text, **kwargs):
        res = super().forward(ecg=ecg, text=text, **kwargs)
        x = res["x"]
        padding_mask = res["padding_mask"]

        x = self.final_dropout(x)

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0

        ecg_feats = x[:, :-text.size(-1)]
        ecg_feats = torch.div(ecg_feats.sum(dim=1), (ecg_feats != 0).sum(dim=1))

        text_feats = x[:, -text.size(-1):]
        text_feats = torch.div(text_feats.sum(dim=1), (text_feats != 0).sum(dim=1))

        x = torch.cat([ecg_feats, text_feats], dim=1)
        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": x,
        }