import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.ecg_transformer import (
    ECGTransformerModel,
    ECGTransformerFinetuningConfig,
)

from fairseq_signals.utils import utils

@dataclass
class DeafTransformerQAConfig(ECGTransformerFinetuningConfig):
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

@register_model("deaf_transformer_qa", dataclass=DeafTransformerQAConfig)
class DeafTransformerQAModel(ECGTransformerModel):
    def __init__(self, cfg: DeafTransformerQAConfig):
        super().__init__(cfg)
        assert cfg.num_ecgs in [1, 2], cfg.num_ecgs
        self.num_ecgs = cfg.num_ecgs
        if self.num_ecgs > 1:
            self.sep_embedding = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )
        
        self.proj = nn.Linear(cfg.encoder_embed_dim * cfg.num_ecgs, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def get_logits(self, net_output, normalize=False, **kwargs):
        logits = net_output["out"]
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits

    def get_targets(self, sample, net_output, **kwargs):
        return sample["answer"].float()

    def forward(
        self,
        ecg,
        ecg_padding_mask=None,
        ecg_2=None,
        ecg_2_padding_mask=None,
        **kwargs
    ):
        feats, ecg_padding_mask = self.get_embeddings(ecg, ecg_padding_mask)
        if ecg_2 is not None:
            assert hasattr(self, "sep_embedding"), (
                "you should initialize `sep_embedding` for processing more than one ecg"
            )
            bsz, tsz = feats.size(0), feats.size(1)

            ecg_2_feats, ecg_2_padding_mask = self.get_embeddings(ecg_2, ecg_2_padding_mask)
            sep_emb = self.sep_embedding.repeat((len(ecg_2_feats), 1, 1))
            ecg_2_feats = torch.cat([sep_emb, ecg_2_feats], dim=1)
            feats = torch.cat([feats, ecg_2_feats], dim=1)

            if ecg_2_padding_mask is not None and ecg_2_padding_mask.any():
                sep_padding_mask = ecg_2_padding_mask.new_zeros((len(ecg_2_padding_mask), 1,))
                sep_padding_mask[torch.where(ecg_2_padding_mask.all(dim=-1))] = True
                if ecg_padding_mask is None:
                    ecg_padding_mask = ecg_2_padding_mask.new_zeros((bsz, tsz))
                padding_mask = torch.cat(
                    [
                        ecg_padding_mask,
                        sep_padding_mask,
                        ecg_2_padding_mask
                    ], dim=1
                )
            else:
                assert ecg_padding_mask is None or (not ecg_padding_mask.any())
                padding_mask = None
        else:
            padding_mask = ecg_padding_mask

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            input_lengths = (1 - ecg_padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:, 0]
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            output_length = output_lengths.max()
        else:
            input_lengths = ecg.new_full((len(ecg),), ecg.shape[-1])
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            output_length = output_lengths.max()
        
        x = self.get_output(feats, padding_mask)["x"]

        feats = x[:, :output_length]
        feats = torch.div(feats.sum(dim=1), (feats != 0).sum(dim=1))
        
        if ecg_2 is not None:
            feats_2 = x[:, output_length:]
            feats_2 = torch.div(feats_2.sum(dim=1), (feats_2 != 0).sum(dim=1) + 1e-8)
            feats = torch.cat([feats, feats_2], dim=1)
        else:
            feats = torch.cat(
                [feats, feats.new_zeros(feats.shape)], dim=1
            )

        x = self.proj(feats)
        
        return {
            "out": x,
            "padding_mask": padding_mask,
        }