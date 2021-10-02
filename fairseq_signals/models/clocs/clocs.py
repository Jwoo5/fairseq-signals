from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import II

import math

import torch
import torch.nn as nn

from fairseq_signals import tasks
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass import Dataclass, ChoiceEnum
from fairseq_signals.models import BaseModel, register_model
from fairseq_signals.models.wav2vec2 import Wav2Vec2Config

from fairseq_signals.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    LayerNorm,
    TransposeLast,
)

CLOCS_MODE_CHOICES = ChoiceEnum(["cmsc", "cmlc", "cmsmlc"])

@dataclass
class ClocsConfig(Dataclass):
    encoder_mode: str = field(
        default = "default",
        metadata= {
            "help": "mode for encoder. default uses conv1d blocks as encoder, "
                    "whereas transformer uses transformer encoder as encoder"
        }
    )
    extractor_mode: str = field(
        default = "default",
        metadata = {
            "help": "mode for conv encoder. default has batch norms in every block, "
                    "whereas group norm has a single group norm with d groups in the "
                    "first conv block, and layer norm has layer norms in every block"
                    "used when encoder_mode == default (conv encoder)"
        }
    )

    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the encoder"}
    )

    conv_layers: str = field(
        default="[(4, 7, 3)] + [(16, 7, 3)] + [(32, 7, 3)]",
        metadata={
            "help": "string describing convolutional layers in form of a python list that contains "
                    "[(dim, kernel_size, stride), ...]"
                    "only used when encoder_mode == default,"
        }
    )

    in_d: int = field(
        default = 12,
        metadata = {"help": "input dimension"}
    )
    sample_size: int = field(
        default = 2500, metadata = {"help": "fixed length of input samples"}
    )
    w2v_path: Optional[str] = field(
        default=None, metadata={"help":"path to wav2vec 2.0 model"}
    )
    apply_mask: bool = False
    data: str = II("task.data")
    # this holds the wav2vec2 args for transformer encoder to override
    w2v_args: Any = None

@register_model("clocs", dataclass = ClocsConfig)
class ClocsModel(BaseModel):
    def __init__(self, cfg: ClocsConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.encoder_mode in {"default", "transformer"}

        conv_layers = eval(cfg.conv_layers)
        if cfg.encoder_mode == "default":
            self.encoder = ConvEncoder(
                conv_layers=conv_layers,
                input_size=cfg.sample_size,                
                in_d=cfg.in_d,
                encoder_embed_dim=cfg.encoder_embed_dim,
                dropout=cfg.dropout,
                mode=cfg.extractor_mode,
                conv_bias=True
            )
        else:
            self.encoder = TransformerEncoder(cfg)
            
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions."""
        return state_dict
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        return logits
    
    def forward(self, source, patient_id=None, segment=None, **kwargs):
        if len(source.shape) < 3:
            source = source.unsqueeze(1)

        x = self.encoder(source, **kwargs)

        x['patient_id'] = patient_id
        x['segment'] = segment
        return x

#XXX deprecated
class ConvEncoder(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        input_size: int,
        in_d: int = 1,
        encoder_embed_dim: int = 256,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm", "group_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_batch_norm = False,
            is_layer_norm = False,
            is_group_norm = False,
            conv_bias = True
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride = stride, bias = conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            assert (
                is_layer_norm and is_group_norm and is_batch_norm
            ) == False, "layer norm, group norm, and batch norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, dim, affine = True),
                        TransposeLast()
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(p=dropout),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    Fp32GroupNorm(dim, dim, affine = True),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(p=dropout)
                )
            elif is_batch_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.BatchNorm1d(dim, affine = True),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(p=dropout)
                )
            else:
                nn.Sequential(make_conv(), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(p=dropout))
        
        self.conv_layers = nn.ModuleList()
        self.output_size = input_size
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm = mode == "layer_norm",
                    is_group_norm = mode == "group_norm" and i == 0,
                    is_batch_norm = mode == "default",
                    conv_bias = conv_bias,
                )
            )
            in_d = dim
            self.output_size = math.floor((self.output_size - k) / stride + 1)
            self.output_size = math.floor((self.output_size - 2) / 2 + 1)
        self.output_size *= dim
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.output_size, encoder_embed_dim),
        )
    
    def forward(self, source, **kwargs):
        x = source
        # B x T -> B x C x T
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        x = self.proj(x)

        return {
            "encoder_out": x, # bsz x n_leads x fsz
        }

class TransformerEncoder(BaseModel):
    def __init__(self, cfg: ClocsConfig):
        super().__init__()
        self.apply_mask = cfg.apply_mask
        assert cfg.w2v_path or cfg.w2v_args

        override_args = {
            "encoder_embed_dim": cfg.encoder_embed_dim,
            "in_d": cfg.in_d
        }


        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, override_args)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        model.remove_pretraining_modules()

        self.w2v_model = model
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask=None, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training
        }

        res = self.w2v_model.extract_features(**w2v_args)
        
        x = res["x"]
        padding_mask = res["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
        
        # (bsz x csz, seq, dim) -- mean-> (bsz x csz, dim)
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        return {
            "encoder_out": x,
            "padding_mask": padding_mask
        }
    
    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict