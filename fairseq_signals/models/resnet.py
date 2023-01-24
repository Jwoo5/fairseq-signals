import contextlib
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals import tasks
from fairseq_signals.utils import utils
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models import register_model
from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task

logger = logging.getLogger(__name__)

@dataclass
class Nejedly2021ResnetConfig(PretrainingConfig):
    """Resnet configs for Nejedly2021Resnet(https://pubmed.ncbi.nlm.nih.gov/35381586/)"""
    num_blocks: int = field(
        default=5, metadata={"help": "number of residual blocks"}
    )
    dim: int = field(
        default=256, metadata={"help": "output dimension for each conv layer"}
    )
    kernel_size: str = field(
        default="[15, 9]",
        metadata={
            "help": "string describing kernerl sizes for the first and the subsequent conv layers "
            "in form of a python list that contains [1st_kernel_size, other_kernel_size]"
        }
    )
    stride: int = field(
        default=2, metadata={"help": "stride for each relevant conv layer"}
    )
    in_d: int = field(
        default=12,
        metadata={"help": "input dimension"}
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )

@register_model("nejedly2021resnet", dataclass=Nejedly2021ResnetConfig)
class Nejedly2021ResnetModel(PretrainingModel):
    """Resnet model for Nejedly2021Resnet(https://pubmed.ncbi.nlm.nih.gov/35381586/)"""
    def __init__(self, cfg: Nejedly2021ResnetConfig):
        super().__init__(cfg)
        k_0, k_1 = eval(cfg.kernel_size)
        self.conv = nn.Conv1d(
            in_channels=cfg.in_d,
            out_channels=cfg.dim,
            kernel_size=k_0,
            padding=(k_0-1) // 2,
            stride=cfg.stride,
            bias=cfg.conv_bias
        )
        self.bn = nn.BatchNorm1d(cfg.dim)
        self.blocks = nn.ModuleList([
            Nejedly2021ResidualBlock(
                d=cfg.dim,
                k=k_1,
                s=cfg.stride,
                downsample=True,
                conv_bias=cfg.conv_bias
            ) for _ in range(cfg.num_blocks)
        ])

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, source, **kwargs):
        x = F.leaky_relu(self.bn(self.conv(source)))

        for block in self.blocks:
            x = block(x)

        return {"x": x}

    def extract_features(self, source):
        res = self.forward(source)
        return res

    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["x"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0
        
        if aggregate:
            logits = torch.div(logits.sum(dim=2), (logits != 0).sum(dim=2))
        
        if normalize:
            logits = utils.log_softmax(logits.float(), dim=1)
        
        return logits

    def get_targets(self, net_output):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: Nejedly2021ResnetConfig,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.Nejedly2021ResnetModel` from a pre-trained model
        checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (Nejedly2021ResnetConfig): cfg to override some arguments of pre-trained model
        """
        state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
        args = state.get("cfg", None)
        if args is None:
            args = convert_namespace_to_omegaconf(state["args"])
        args.criterion = None
        args.lr_scheduler = None

        assert cfg.normalize == args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )
        assert cfg.filter == args.task.filter, (
            "Fine-tuning works best when signal filtering for data is the same. "
            "Please check that --filter is set or unset for both pre-training and here"
        )

        args.task.data = cfg.data
        task = tasks.setup_task(args.task)
        model = task.build_model(args.model)

        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class Nejedly2021ResnetFinetuningConfig(FinetuningConfig, Nejedly2021ResnetConfig):
    final_dropout: float = field(
        default=0.5,
        metadata={"help": "dropout after resnet and before final projection"}
    )

class Nejedly2021ResnetFinetuningModel(FinetuningModel):
    def __init__(self, cfg: Nejedly2021ResnetFinetuningConfig, encoder: Nejedly2021ResnetModel):
        super().__init__(cfg, encoder)

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: Nejedly2021ResnetFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = Nejedly2021ResnetModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = Nejedly2021ResnetModel(cfg)

        return cls(cfg, encoder)

    def forward(self, source, **kwargs):
        args = {"source": source}

        ft = self.freeze_finetune_updates <= self.num_updates
        
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)
        
        return res

class Nejedly2021ResidualBlock(nn.Module):
    def __init__(self, d=256, k=9, s=2, downsample=False, conv_bias=False):
        super().__init__()
        self.downsample = downsample
        self.stride = s if downsample else 1
        
        p = (k - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=k,
            stride=self.stride,
            padding=p,
            bias=conv_bias
        )
        self.bn1 = nn.BatchNorm1d(d)
        self.conv2 = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=k,
            padding=p,
            bias=conv_bias
        )
        self.bn2 = nn.BatchNorm1d(256)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride, ceil_mode=True)
            self.idfunc_1 = nn.Conv1d(
                in_channels=d,
                out_channels=d,
                kernel_size=1,
                bias=conv_bias
            )
    
    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
        
        x = x + identity
        return x
