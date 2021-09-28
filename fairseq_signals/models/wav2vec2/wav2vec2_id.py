import torch
import torch.nn as nn

from dataclasses import dataclass, field
from fairseq_signals.models import BaseModel, register_model
from fairseq_signals.models.wav2vec2 import Wav2Vec2DcModel, Wav2Vec2DcConfig
from fairseq_signals.utils import utils

@dataclass
class Wav2Vec2IDConfig(Wav2Vec2DcConfig):
    pass


@register_model("wav2vec2_id", dataclass=Wav2Vec2IDConfig)
class Wav2Vec2IDModel(Wav2Vec2DcModel):
    def __init__(self, cfg: Wav2Vec2IDConfig, w2v_encoder: BaseModel):
        super().__init__(cfg, w2v_encoder)
        self.w2v_encoder.proj = None

        self.kernel = nn.Parameter(
            torch.Tensor(
                cfg.output_size,
                cfg.w2v_args.model.encoder_embed_dim,
            )
        )
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def get_logits(self, net_output, normalize = False, aggregate = False):
        logits = net_output["encoder_out"]

        # TODO need to be checked whether to work properly
        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        # TODO aggregate over tokens to classify the whole outputs
        # example: logits = net_output["encoder_out"].mean(1).float() # B x T x n_classes -> B x n_classes
        #       ... mean is too naive
        if aggregate:
            # logits = torch.div(logits.sum(dim = 1), (logits != 0).sum(dim = 1))
            pass
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim = -1)

        return logits

    def get_targets(self, sample, net_output):
        return sample["label"].long()

    def forward(self, **kwargs):
        net_output = self.w2v_encoder(**kwargs)
        x = net_output["encoder_out"]
        x = torch.div(x.sum(dim = 1), (x != 0).sum(dim = 1))

        norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm)

        net_output["encoder_out"] = x

        return net_output

    def get_cosine_similarity(self, logits):
        norm = torch.norm(self.kernel, dim=1, keepdim=True)
        weights = torch.div(self.kernel, norm)

        return torch.mm(logits, weights.T).clamp(-1,1)