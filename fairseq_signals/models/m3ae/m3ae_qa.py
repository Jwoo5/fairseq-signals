from dataclasses import dataclass, field
from omegaconf import MISSING

import torch
import torch.nn as nn

from fairseq_signals.models import register_model
from fairseq_signals.models.m3ae import (
    init_weights,
    M3AEModel,
    M3AEFinetuningConfig,
    M3AEFinetuningModel
)

from fairseq_signals.utils import utils

@dataclass
class M3AEQAConfig(M3AEFinetuningConfig):
    num_labels: int = field(
        default=MISSING,
        metadata={
            "help": "number of classes (answers) in the dataset"
        }
    )
    num_ecgs: int = field(
        default=1,
        metadata={
            "help": "number of ecgs to be processed at a time. 1 or 2 allowed only"
        }
    )

@register_model("m3ae_qa", dataclass=M3AEQAConfig)
class M3AEQAModel(M3AEFinetuningModel):
    def __init__(self, cfg: M3AEQAConfig, encoder: M3AEModel):
        super().__init__(cfg, encoder)
        assert cfg.num_ecgs in [1, 2], cfg.num_ecgs
        self.num_ecgs = cfg.num_ecgs
        if self.num_ecgs > 1:
            sep_embedding = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )
            self.encoder.__setattr__("sep_embedding", sep_embedding)
        
        self.head = nn.Sequential(
            nn.Linear(cfg.encoder_embed_dim * 2, cfg.encoder_embed_dim * 2),
            nn.LayerNorm(cfg.encoder_embed_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.encoder_embed_dim * 2, cfg.num_labels)
        )
        self.head.apply(init_weights)
    
    def get_logits(self, net_output, normalize=False):
        logits = net_output["out"]

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits

    def get_targets(self, sample, net_output):
        return sample["answer"].float()

    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask=None,
        text_padding_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        **kwargs
    ):
        if text_padding_mask is not None:
            text_attention_mask = ~text_padding_mask
        else:
            text_attention_mask = text.new_ones(text.shape).bool()

        res = super().forward(
            ecg=ecg,
            text=text,
            ecg_padding_mask=ecg_padding_mask,
            text_attention_mask=text_attention_mask,
            ecg_2=ecg_2,
            ecg_2_padding_mask=ecg_2_padding_mask,
            mask=False,
            **kwargs
        )
        
        x = self.head(res["multi_modal_cls_feats"])
        return {
            "out": x,
        }