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
class SEWideResidualNetworkConfig(PretrainingConfig):
    """Implementation of SE-WRN which is similar to https://iopscience.iop.org/article/10.1088/1361-6579/acb30f"""
    num_blocks: int = field(
        default=3, metadata={"help": "number of residual blocks for each stage"}
    )
    num_filters: str = field(
        default="[16, 16, 32, 64, 128]",
        metadata={
            "help": "string describing number of filters for the input stem and each stage"
            " in form of a python list"
        }
    )
    widening_factor: int = field(
        default=1,
        metadata={
            "help": "widening factor multiplies the number of filters in conv layers"
        }
    )
    kernel_size: int = field(
        default=11,
        metadata={
            "help": "kernel size for each convolutional layer"
        }
    )
    in_d: int = field(
        default=12,
        metadata={"help": "input dimension"}
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    ratio: int = field(
        default=8, metadata={"help": "ratio for SE module, which divides the number of channels"}
    )
    leadwise: bool = field(
        default=False, metadata={
            "help": "whether to apply convolution filter lead-wisely. "
            "If set, in_d should be 1"
        }
    )

@register_model("se_wrn", dataclass=SEWideResidualNetworkConfig)
class SEWideResidualNetworkModel(PretrainingModel):
    """Implementation of SE-WRN which is similar to https://iopscience.iop.org/article/10.1088/1361-6579/acb30f"""
    def __init__(self, cfg: SEWideResidualNetworkConfig):
        super().__init__(cfg)
        self.leadwise = cfg.leadwise
        self.num_filters = eval(cfg.num_filters)
        if self.leadwise:
            assert cfg.in_d == 1, (
                "--in_d should be 1 when --leadwise is set to True"
            )
            self.conv = nn.Conv2d(
                in_channels=cfg.in_d,
                out_channels=self.num_filters[0],
                kernel_size=(1, cfg.kernel_size),
                padding=(0, (cfg.kernel_size - 1) // 2),
                stride=(1, 1),
                bias=cfg.conv_bias
            )
            self.bn = nn.BatchNorm2d(self.num_filters[0])
        else:
            self.conv = nn.Conv1d(
                in_channels=cfg.in_d,
                out_channels=self.num_filters[0],
                kernel_size=cfg.kernel_size,
                padding=(cfg.kernel_size - 1) // 2,
                stride=1,
                bias=cfg.conv_bias
            )
            self.bn = nn.BatchNorm1d(self.num_filters[0])
        self.relu = nn.ReLU()
        self.groups = nn.ModuleList([
            SEWideResidualGroup(
                in_channels=self.num_filters[i],
                out_channels=self.num_filters[i+1],
                kernel_size=cfg.kernel_size,
                num_blocks=cfg.num_blocks,
                ratio=cfg.ratio,
                widening_factor=cfg.widening_factor,
                conv_bias=cfg.conv_bias,
                is_first= i == 0,
                leadwise=cfg.leadwise
            ) for i in range(len(self.num_filters) - 1)
        ])

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, source, **kwargs):
        bsz, csz, tsz = source.shape
        if self.leadwise:
            source = source.view(bsz, 1, csz, tsz)
        x = self.relu(self.bn(self.conv(source)))

        for stage in self.groups:
            x = stage(x)

        if self.leadwise:
            x = x.transpose(1,2).contiguous()

        return {"x": x}

    def extract_features(self, source):
        res = self.forward(source)
        return res

    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["x"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0

        if aggregate:
            logits = torch.div(logits.sum(dim=-1), (logits != 0).sum(dim=-1))

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=1)

        return logits

    def get_targets(self, net_output):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: SEWideResidualNetworkConfig,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.SEWideResidualNetworkModel` from a pre-trained model
        checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (SEWideResidualNetworkConfig): cfg to override some arguments of pre-trained model
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
        task = tasks.setup_task(args.task, from_checkpoint=True)
        model = task.build_model(args.model)

        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()

        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class SEWideResidualNetworkFinetuningConfig(FinetuningConfig, SEWideResidualNetworkConfig):
    final_dropout: float = field(
        default=0.3,
        metadata={"help": "dropout after resnet and before final projection"}
    )

class SEWideResidualNetworkFinetuningModel(FinetuningModel):
    def __init__(self, cfg: SEWideResidualNetworkFinetuningConfig, encoder: SEWideResidualNetworkModel):
        super().__init__(cfg, encoder)

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: SEWideResidualNetworkFinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = SEWideResidualNetworkModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = SEWideResidualNetworkModel(cfg)

        return cls(cfg, encoder)

    def forward(self, source, **kwargs):
        args = {"source": source}

        ft = self.freeze_finetune_updates <= self.num_updates
        
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)
        
        return res

class SEWideResidualGroup(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_blocks,
        ratio=8,
        widening_factor=1,
        conv_bias=False,
        is_first=False,
        leadwise=False,
    ):
        super().__init__()
        assert (out_channels * widening_factor) % ratio == 0

        modules = [
            SEWideResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                ratio=ratio,
                widening_factor=widening_factor,
                dropout=0.3,
                downsample=True,
                conv_bias=False,
                is_first=is_first,
                leadwise=leadwise
            )
        ]
        for _ in range(num_blocks - 1):
            modules.append(
                SEWideResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    ratio=ratio,
                    widening_factor=widening_factor,
                    dropout=0.3,
                    downsample=False,
                    conv_bias=False,
                    is_first=False,
                    leadwise=leadwise
                )
            )
        
        self.layers = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class SEWideResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=2,
        ratio=8,
        widening_factor=1,
        dropout=0.3,
        downsample=False,
        conv_bias=False,
        is_first=False,
        leadwise=False
    ):
        super().__init__()
        assert (out_channels * widening_factor) % ratio == 0
        
        self.leadwise = leadwise
        self.downsample = downsample
        self.stride = stride if downsample else 1
        padding = (kernel_size - 1) // 2

        self.in_channels = in_channels * widening_factor
        self.out_channels = out_channels * widening_factor

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = None
        if not is_first:
            if leadwise:
                self.bn1 = nn.BatchNorm2d(self.in_channels)
            else:
                self.bn1 = nn.BatchNorm1d(self.in_channels)
        
        if leadwise:
            self.conv1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, kernel_size),
                stride=(1, self.stride),
                padding=(0, padding),
                bias=conv_bias
            )
            self.bn2 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                bias=conv_bias
            )
            self.gap = nn.AdaptiveAvgPool2d(output_size=(12, 1)) # (64, 256, 12)
        else:
            self.conv1 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                bias=conv_bias
            )
            self.bn2 = nn.BatchNorm1d(self.out_channels)
            self.conv2 = nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=conv_bias
            )

            self.gap = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc1 = nn.Linear(out_channels, out_channels // ratio)
        self.fc2 = nn.Linear(out_channels // ratio, out_channels)
        self.dropout = nn.Dropout(dropout)

        if self.downsample:
            if leadwise:
                self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, self.stride), stride=(1, self.stride), ceil_mode=True)
                self.idfunc_1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=(1, 1),
                    bias=conv_bias
                )
            else:
                self.idfunc_0 = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride, ceil_mode=True)
                self.idfunc_1 = nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=conv_bias
                )

    def forward(self, x):
        identity = x
        if self.bn1 is not None:
            x = self.relu(self.bn1(x))

        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.conv2(x)

        se = self.gap(x).squeeze(-1)
        if self.leadwise:
            se = se.transpose(1, 2)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)
        if self.leadwise:
            se = se.transpose(1, 2)
        x = x * se

        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)

        x = x + identity
        return x