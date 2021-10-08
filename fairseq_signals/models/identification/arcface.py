import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from fairseq_signals.models import BaseModel, register_model
from fairseq_signals.models.conv_transformer import (
    MASKING_DISTRIBUTION_CHOICES,
    ConvTransformerModel,
    ConvTransformerConfig
)
from fairseq_signals.utils import utils
from fairseq_signals.tasks import Task


@dataclass
class ArcFaceConfig(ConvTransformerConfig):
    model_path: Optional[str] = field(
        default=None, metadata={"help": "path to pretrained model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    freeze_finetune_updates: int = field(
        default = 0, metadata = {"help": "dont finetune wav2vec2 for this many updates"}
    )
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )

    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )

    # overriding arguments
    dropout: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    mask_length: int = 10
    mask_prob: float = 0.5
    mask_selection: MASKING_DISTRIBUTION_CHOICES = "static"
    mask_other: float = 0
    no_mask_overlap: bool = False
    mask_channel_length: int = 10
    mask_channel_prob: float = 0.0
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = "static"
    mask_channel_other: float = 0
    no_mask_channel_overlap: bool = False
    encoder_layerdrop: float = 0.0
    feature_grad_mult: float = 0.0

    output_size: int = II("task.num_labels")

@register_model("arcface", dataclass=ArcFaceConfig)
class ArcFaceModel(BaseModel):
    def __init__(self, cfg: ArcFaceConfig, encoder: ConvTransformerModel):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

        if not cfg.apply_mask:
            if hasattr(self.encoder, "mask_emb"):
                self.encoder.mask_emb = None

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.kernel = nn.Parameter(
            torch.Tensor(
                cfg.output_size,
                cfg.encoder_embed_dim
            )
        )
        self.kenrel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: ArcFaceConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = ConvTransformerModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = ConvTransformerModel(cfg)
        
        return cls(cfg, encoder)
    
    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["encoder_out"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        if aggregate:
            logits = torch.div(logits.sum(dim=1), (logits != 0).sum(dim=1))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits
    
    def get_targets(self, sample, net_output):
        return sample["label"].float()
    
    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)
    
    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder(**kwargs)

            x = res["x"]
            padding_mask = res["padding_mask"]
        
        x = self.final_dropout(x)
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        # if self.cfg.in_d == 1:
        #     concat
        #     ...

        norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm)

        return {
            "encoder_out": x,
            "padding_mask": padding_mask
        }
    
    def get_cosine_similarity(self, logits):
        norm = torch.norm(self.kernel, dim=1, keepdim=True)
        weights = torch.div(self.kernel, norm)

        return torch.mm(logits, weights.T).clamp(-1,1)