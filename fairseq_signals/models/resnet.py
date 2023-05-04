import contextlib
from dataclasses import dataclass, field
from typing import Type, Union, Optional, Callable, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals import tasks
from fairseq_signals.utils import utils
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass import ChoiceEnum
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models import register_model
from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task

logger = logging.getLogger(__name__)

RESNET_CFG_CHOICES = ChoiceEnum(["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])

@dataclass
class VanillaResnetConfig(PretrainingConfig):
    """Vanilla Resnet1D configs"""
    configuration: RESNET_CFG_CHOICES = field(
        default="resnet50",
        metadata={"help": "choice for resnet configuration"}
    )
    in_d: int = field(
        default=12,
        metadata={"help": "input dimension"}
    )

class VanillaResnetModel(PretrainingModel):
    """
    Vanilla Resnet1D implementation.
    Most of codes are borrowed from torchvision.models.resnet
    (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
    but some modifications are added such as using nn.Conv1d instead of nn.Conv2d
    and getting rid of the classifier head.
    """
    def __init__(self, cfg: VanillaResnetConfig):
        super().__init__(cfg)
        self.network_block = {
            "resnet18": BasicBlock,
            "resnet34": BasicBlock,
            "resnet50": Bottleneck,
            "resnet101": Bottleneck,
            "resnet152": Bottleneck
        }
        self.network_layers = {
            "resnet18": [2, 2, 2, 2],
            "resnet34": [3, 4, 6, 3],
            "resnet50": [3, 4, 6, 3],
            "resnet101": [3, 4, 23, 3],
            "resnet152": [3, 8, 36, 3]
        }
        self.network_choice = cfg.configuration

        self.block = self.network_block[self.network_choice]
        self.layers = self.network_layers[self.network_choice]

        self.featuare_extractor = Resnet(self.block, self.layers, cfg.in_d)

        self.num_updates= 0
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, source, normalize=False, **kwargs):
        x = self.featuare_extractor(source)
        x = torch.flatten(x, 1)
        
        if normalize:
            x = F.normalize(x, dim=1)

        return {"x": x}

    def extract_features(self, source):
        res = self.forward(source)
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]

        return logits
    
    def get_targets(self, net_output):
        raise NotImplementedError()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: VanillaResnetConfig,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.VanillaResnetModel` from a pre-trained model checkpoint.

        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (VanillaResnetConfig): cfg to override some arguments of pre-trained model
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

        if hasattr(model, "remove_pretrainined_modules"):
            model.remove_pretraining_modules()
        
        model.load_state_dict(state["model"], strict=True)
        logger.info(f"Loaded pre-trained model parameters from {model_path}")

        return model

@dataclass
class VanillaResnetFinetuningConfig(FinetuningConfig, VanillaResnetConfig):
    pass

class VanillaResnetFinetuningModel(FinetuningModel):
    def __init__(self, cfg: VanillaResnetConfig, encoder: VanillaResnetModel):
        super().__init__(cfg, encoder)

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
    
    @classmethod
    def build_model(cls, cfg: FinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = VanillaResnetModel.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = VanillaResnetModel(cfg)
        
        return cls(cfg, encoder)

    def forward(self, source, **kwargs):
        args = {"source": source}

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)
        
        return res

def conv1x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Callable[..., nn.Module]] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_d: int = 12,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_d, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

@dataclass
class Nejedly2021ResnetConfig(PretrainingConfig):
    """Resnet configs for Nejedly2021Resnet(https://pubmed.ncbi.nlm.nih.gov/35381586/)"""
    num_blocks: int = field(
        default=5, metadata={"help": "number of residual blocks"}
    )
    num_filters: int = field(
        default=256, metadata={"help": "number of filters for each conv layer"}
    )
    kernel_size: str = field(
        default="[15, 9]",
        metadata={
            "help": "string describing kernel sizes for the first and the subsequent conv layers "
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
    leadwise: bool = field(
        default=False, metadata={
            "help": "whether to apply convolution filter lead-wisely. "
            "If set, in_d should be 1"
        }
    )

@register_model("nejedly2021resnet", dataclass=Nejedly2021ResnetConfig)
class Nejedly2021ResnetModel(PretrainingModel):
    """Resnet model for Nejedly2021Resnet(https://pubmed.ncbi.nlm.nih.gov/35381586/)"""
    def __init__(self, cfg: Nejedly2021ResnetConfig):
        super().__init__(cfg)
        self.leadwise = cfg.leadwise
        if self.leadwise:
            assert cfg.in_d == 1, (
                "--in_d should be 1 when --leadwise is set to True"
            )

        k_0, k_1 = eval(cfg.kernel_size)
        if self.leadwise:
            self.conv = nn.Conv2d(
                in_channels=cfg.in_d,
                out_channels=cfg.num_filters,
                kernel_size=(1, k_0),
                padding=(0, (k_0 - 1) // 2),
                stride=(1, cfg.stride),
                bias=cfg.conv_bias
            )
            self.bn = nn.BatchNorm2d(cfg.num_filters)
        else:
            self.conv = nn.Conv1d(
                in_channels=cfg.in_d,
                out_channels=cfg.num_filters,
                kernel_size=k_0,
                padding=(k_0 - 1) // 2,
                stride=cfg.stride,
                bias=cfg.conv_bias
            )
            self.bn = nn.BatchNorm1d(cfg.num_filters)

        self.blocks = nn.ModuleList([
            Nejedly2021ResidualBlock(
                d=cfg.num_filters,
                k=k_1,
                s=cfg.stride,
                downsample=True,
                conv_bias=cfg.conv_bias,
                leadwise=self.leadwise,
            ) for _ in range(cfg.num_blocks)
        ])


    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, source, **kwargs):
        bsz, csz, tsz = source.shape
        l = (source != 0).all(dim=-1).to(source.dtype)
        if self.leadwise:
            source = source.view(bsz, 1, csz, tsz)
        x = F.leaky_relu(self.bn(self.conv(source)))

        for block in self.blocks:
            x = block(x)

        if self.leadwise:
            x = x.view(bsz, csz, self.cfg.num_filters, -1)

        return {"x": x, "l": l}

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
        task = tasks.setup_task(args.task, from_checkpoint=True)
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
    def __init__(self, d=256, k=9, s=2, downsample=False, conv_bias=False, leadwise=False):
        super().__init__()
        self.leadwise = leadwise
        self.downsample = downsample
        self.stride = s if downsample else 1
        
        p = (k - 1) // 2
        if leadwise:
            self.conv1 = nn.Conv2d(
                in_channels=d,
                out_channels=d,
                kernel_size=(1, k),
                stride=(1, self.stride),
                padding=(0, p),
                bias=conv_bias
            )
            self.bn1 = nn.BatchNorm2d(d)
            self.conv2 = nn.Conv2d(
                in_channels=d,
                out_channels=d,
                kernel_size=(1, k),
                padding=(0, p),
                bias=conv_bias
            )
            self.bn2 = nn.BatchNorm2d(d)
        else:
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
            self.bn2 = nn.BatchNorm1d(d)

        if self.downsample:
            if leadwise:
                self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, self.stride), stride=(1, self.stride), ceil_mode=True)
                self.idfunc_1 = nn.Conv2d(
                    in_channels=d,
                    out_channels=d,
                    kernel_size=(1, 1),
                    bias=conv_bias
                )
            else:
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