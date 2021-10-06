import torch
import torch.nn as nn

from dataclasses import dataclass, field
from fairseq_signals.models import BaseModel, register_model
from fairseq_signals.models.clocs import ClocsDcModel, ClocsDcConfig
from fairseq_signals.utils import utils

@dataclass
class ClocsIDConfig(ClocsDcConfig):
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    layerdrop: float = field(
        default = 0.0, metadata = {"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    activation_dropout: float = field(
        default = 0.0,
        metadata = {
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        }
    )
    feature_grad_mult: float = field(
        default = 0.0, metadata = {"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )

@register_model("clocs_id", dataclass=ClocsIDConfig)
class ClocsIDModel(ClocsDcModel):
    def __init__(self, cfg: ClocsIDConfig, clocs_encoder: BaseModel):
        super().__init__(cfg, clocs_encoder)
        self.clocs_encoder.proj = None

        #XXX temp
        self.clocs_encoder.clocs_model.encoder.w2v_model.feature_grad_mult = cfg.feature_grad_mult
        self.clocs_encoder.clocs_model.encoder.w2v_model.encoder_layerdrop = cfg.layerdrop
        self.clocs_encoder.clocs_model.encoder.w2v_model.dropout = cfg.dropout
        self.clocs_encoder.clocs_model.encoder.w2v_model.activation_dropout = cfg.activation_dropout

        self.kernel = nn.Parameter(
            torch.Tensor(
                cfg.output_size,
                cfg.clocs_args.model.encoder_embed_dim
            )
        )
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["encoder_out"]
        return logits
    
    def get_targets(self, sample, net_output):
        return sample["label"].long()
    
    def forward(self, **kwargs):
        net_output = self.clocs_encoder(**kwargs)
        x = net_output["encoder_out"]

        norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm)

        net_output["encoder_out"] = x

        return net_output
    
    def get_cosine_similarity(self, logits):
        norm = torch.norm(self.kernel, dim=1, keepdim=True)
        weights = torch.div(self.kernel, norm)

        return torch.mm(logits, weights.T).clamp(-1,1)