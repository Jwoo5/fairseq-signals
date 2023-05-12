import torch
import torch.nn as nn
import torch.nn.functional as F

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
        metadata={
            "help": "number of classes in the dataset"
        }
    )
    num_ecgs: int = field(
        default=1,
        metadata={
            "help": "number of ecgs to be processed at a time. 1 or 2 allowed only."
        }
    )

@register_model("ecg_language_transformer_classifier", dataclass=ECGLanguageTransformerClassificationConfig)
class ECGLanguageTransformerClassificationModel(ECGLanguageTransformerFinetuningModel):
    def __init__(self, cfg: ECGLanguageTransformerClassificationConfig, encoder: ECGLanguageTransformerModel):
        super().__init__(cfg, encoder)
        assert cfg.num_ecgs in [1, 2], cfg.num_ecgs
        self.num_ecgs = cfg.num_ecgs

        self.proj = nn.Linear(cfg.encoder_embed_dim * (cfg.num_ecgs + 1), cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

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
        res = super().forward(
            ecg=ecg,
            text=text,
            ecg_padding_mask=ecg_padding_mask,
            text_padding_mask=text_padding_mask,
            ecg_2=ecg_2,
            ecg_2_padding_mask=ecg_2_padding_mask,
            **kwargs
        )
        x = res["x"]
        padding_mask = res["padding_mask"]

        x = self.final_dropout(x)

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            input_lengths = (1 - ecg_padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:, 0]
            output_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
            output_length = output_lengths.max()
        else:
            input_lengths = x.new_full((len(x),), ecg.shape[-1])
            output_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths)
            output_length = output_lengths.max()

        ecg_feats = x[:, :output_length]
        ecg_feats = torch.div(ecg_feats.sum(dim=1), (ecg_feats != 0).sum(dim=1))

        if ecg_2 is not None:
            if ecg_2_padding_mask is not None and ecg_2_padding_mask.any():
                input_lengths_2 = (1 - ecg_2_padding_mask.long()).sum(-1)
                if input_lengths_2.dim() > 1:
                    for input_len in input_lengths_2:
                        assert (input_len == input_len[0]).all()
                    input_lengths_2 = input_lengths_2[:, 0]
                output_lengths_2 = self.encoder._get_feat_extract_output_lengths(input_lengths_2)
                output_length_2 = output_lengths_2.max()
            else:
                input_lengths_2 = x.new_full((len(x),), ecg_2.shape[-1])
                output_lengths_2 = self.encoder._get_feat_extract_output_lengths(input_lengths_2)
                output_length_2 = output_lengths_2.max()
        
            ecg_feats_2 = x[:, output_length: output_length + output_length_2]
            ecg_feats_2 = torch.div(ecg_feats_2.sum(dim=1), (ecg_feats_2 != 0).sum(dim=1) + 1e-8)

            ecg_feats = torch.cat([ecg_feats, ecg_feats_2], dim=1)
        elif self.num_ecgs == 2:
            ecg_feats = torch.cat([ecg_feats, ecg_feats.new_zeros(ecg_feats.shape)], dim=1)

        text_feats = x[:, -text.size(-1):]
        text_feats = torch.div(text_feats.sum(dim=1), (text_feats != 0).sum(dim=1))

        x = torch.cat([ecg_feats, text_feats], dim=1)
        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": x,
        }