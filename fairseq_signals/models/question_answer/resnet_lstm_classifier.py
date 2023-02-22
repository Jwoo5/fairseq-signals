import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq_signals.models import register_model
from fairseq_signals.models.resnet_lstm import (
    ResnetLSTMModel,
    ResnetLSTMFinetuningConfig,
    ResnetLSTMFinetuningModel
)

from fairseq_signals.utils import utils

@dataclass
class ResnetLSTMClassificationConfig(ResnetLSTMFinetuningConfig):
    num_labels: int = field(
        default=MISSING,
        metadata={
        "help": "number of classes in the dataset"
        }
    )

@register_model("resnet_lstm_classifier", dataclass=ResnetLSTMClassificationConfig)
class ResnetLSTMClassificationModel(ResnetLSTMFinetuningModel):
    def __init__(self, cfg: ResnetLSTMClassificationConfig, encoder: ResnetLSTMModel):
        super().__init__(cfg, encoder)

        self.proj = nn.Linear(cfg.final_dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
    
    def get_logits(self, net_output, normalize=False):
        logits = net_output["out"]

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)
        
        return logits
    
    def get_targets(self, sample, net_output):
        return sample["answer"].float()
    
    def forward(self, ecg, text, text_padding_mask=None, **kwargs):
        res = super().forward(ecg=ecg, text=text, text_padding_mask=text_padding_mask, **kwargs)
        x = res["x"]

        x = self.proj(x)

        return {
            "encoder_out": res["x"].detach(),
            "out": x
        }