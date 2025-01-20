import contextlib
import logging
from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task
from fairseq_signals.dataclass import ChoiceEnum

UNET_BACKBONE_CHOICES = ChoiceEnum(["unet", "unet3plus"])

logger = logging.getLogger(__name__)

@dataclass
class ECGUNetConfig(PretrainingConfig):
    """Unet for ECG signals (1d) configs"""
    in_d: int = field(
        default=1,
        metadata={"help": "input dimension"}
    )
    initial_dim: int = field(
        default=32,
        metadata={"help": "output dimension for the first encoder in unet"}
    )
    kernel_size: int = field(
        default=9,
        metadata={"help": "kernel sizes for all conv layers in unet"}
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    dropout: float = field(
        default=0.2, metadata={"help": "dropout probability for the conv layers in unet"}
    )

class ECGUNetModel(PretrainingModel):
    """Unet for ECG signals (1d)"""
    def __init__(self, cfg: ECGUNetConfig):
        super().__init__(cfg)
        
        self.final_dim = cfg.initial_dim

        all_dims = [cfg.initial_dim * (2 ** n) for n in range(5)]
        self.all_dims = all_dims

        self.down1 = DownBlock(cfg.in_d, all_dims[0], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        self.down2 = DownBlock(all_dims[0], all_dims[1], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        self.down3 = DownBlock(all_dims[1], all_dims[2], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        self.down4 = DownBlock(all_dims[2], all_dims[3], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        
        self.middle = nn.Sequential(
            ConvBnRelu1d(all_dims[3], all_dims[4], cfg.kernel_size, cfg.dropout, cfg.conv_bias),
            ConvBnRelu1d(all_dims[4], all_dims[4], cfg.kernel_size, cfg.dropout, cfg.conv_bias),
        )
        
        self.up4 = UpBlock(
            all_dims[4], all_dims[3], all_dims[3], cfg.kernel_size, cfg.dropout, cfg.conv_bias
        )
        self.up3 = UpBlock(
            all_dims[3], all_dims[2], all_dims[2], cfg.kernel_size, cfg.dropout, cfg.conv_bias
        )
        self.up2 = UpBlock(
            all_dims[2], all_dims[1], all_dims[1], cfg.kernel_size, cfg.dropout, cfg.conv_bias
        )
        self.up1 = UpBlock(
            all_dims[1], all_dims[0], all_dims[0], cfg.kernel_size, cfg.dropout, cfg.conv_bias
        )

    @classmethod
    def build_model(cls, cfg, task: Task = None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, x, **kwargs):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        
        x = self.middle(x)
        
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return {"x": x}

    def extract_features(self, source, **kwargs):
        res = self.forward(source, **kwargs)
        return res

    def get_logits(self, sample, net_output, **kwargs):
        logits = net_output["x"]
        return logits

    def get_targets(self, sample, net_output, **kwargs):
        raise NotImplementedError()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: ECGUNetConfig,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.ECGUNetModel` from a pre-trained model checkpoint.
        
        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (ECGUNetConfig): cfg to override some arguments of pre-trained model
        """
        return super().from_pretrained(model_path, cfg, **kwargs)

@dataclass
class ECGUNet3PlusConfig(ECGUNetConfig):
    pass

class ECGUNet3PlusModel(PretrainingModel):
    """Unet3+ model for ECG signals (1d)"""
    def __init__(self, cfg: ECGUNet3PlusConfig):
        super().__init__(cfg)
        
        
        all_dims = [cfg.initial_dim * (2 ** n) for n in range(5)]
        skip_dim = cfg.initial_dim
        up_dim = skip_dim * 5

        self.all_dims = all_dims
        self.final_dim = up_dim
        
        self.down1 = DownBlock(cfg.in_d, all_dims[0], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        self.down2 = DownBlock(all_dims[0], all_dims[1], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        self.down3 = DownBlock(all_dims[1], all_dims[2], cfg.kernel_size, cfg.dropout, cfg.conv_bias)
        self.down4 = DownBlock(all_dims[2], all_dims[3], cfg.kernel_size, cfg.dropout, cfg.conv_bias)

        self.middle = nn.Sequential(
            ConvBnRelu1d(all_dims[3], all_dims[4], cfg.kernel_size, cfg.dropout, cfg.conv_bias),
            ConvBnRelu1d(all_dims[4], all_dims[4], cfg.kernel_size, cfg.dropout, cfg.conv_bias),
        )
        
        self.up4 = UpBlockForUNet3Plus(
            all_dims, skip_dim, up_dim, cfg.kernel_size, cfg.dropout, cfg.conv_bias
        )
        self.up3 = UpBlockForUNet3Plus(
            all_dims=all_dims[:3] + [up_dim] * 1 + all_dims[4:],
            skip_dim=skip_dim,
            dim=up_dim,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            conv_bias=cfg.conv_bias
        )
        self.up2 = UpBlockForUNet3Plus(
            all_dims=all_dims[:2] + [up_dim] * 2 + all_dims[4:],
            skip_dim=skip_dim,
            dim=up_dim,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            conv_bias=cfg.conv_bias
        )
        self.up1 = UpBlockForUNet3Plus(
            all_dims=all_dims[:1] + [up_dim] * 3 + all_dims[4:],
            skip_dim=skip_dim,
            dim=up_dim,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            conv_bias=cfg.conv_bias
        )

    @classmethod
    def build_model(cls, cfg, task: Task =None):
        """Build a new model instance."""
        return cls(cfg)

    def forward(self, x, return_features=False, **kwargs):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x5 = self.middle(x)

        x4 = self.up4(
            F.max_pool1d(skip1, kernel_size=8, stride=8),
            F.max_pool1d(skip2, kernel_size=4, stride=4),
            F.max_pool1d(skip3, kernel_size=2, stride=2),
            skip4,
            F.interpolate(x5, size=skip4.shape[-1], mode="linear", align_corners=False)
        )
        x3 = self.up3(
            F.max_pool1d(skip1, kernel_size=4, stride=4),
            F.max_pool1d(skip2, kernel_size=2, stride=2),
            skip3,
            F.interpolate(x4, size=skip3.shape[-1], mode="linear", align_corners=False),
            F.interpolate(x5, size=skip3.shape[-1], mode="linear", align_corners=False)
        )
        x2 = self.up2(
            F.max_pool1d(skip1, kernel_size=2, stride=2),
            skip2,
            F.interpolate(x3, size=skip2.shape[-1], mode="linear", align_corners=False),
            F.interpolate(x4, size=skip2.shape[-1], mode="linear", align_corners=False),
            F.interpolate(x5, size=skip2.shape[-1], mode="linear", align_corners=False),
        )
        x1 = self.up1(
            skip1,
            F.interpolate(x2, size=skip1.shape[-1], mode="linear", align_corners=False),
            F.interpolate(x3, size=skip1.shape[-1], mode="linear", align_corners=False),
            F.interpolate(x4, size=skip1.shape[-1], mode="linear", align_corners=False),
            F.interpolate(x5, size=skip1.shape[-1], mode="linear", align_corners=False),
        )

        res = {"x": x1}
        if return_features:
            res["skip1"] = skip1
            res["skip2"] = skip2
            res["skip3"] = skip3
            res["skip4"] = skip4
            res["skip5"] = x5

        return res

    def extract_features(self, source, **kwargs):
        res = self.forward(source, **kwargs)
        return res
    
    def get_logits(self, sample, net_output, **kwargs):
        logits = net_output["x"]
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        cfg: ECGUNet3PlusConfig,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.ECGUNet3PlusModel` from a pre-trained model
        checkpoint.
        
        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (ECGUNet3PlusConfig): cfg to override some arguments of pre-trained model
        """
        return super().from_pretrained(model_path, cfg, **kwargs)

@dataclass
class ECGUNetFinetuningConfig(FinetuningConfig, ECGUNet3PlusConfig):
    backbone: UNET_BACKBONE_CHOICES = field(
        default="unet",
        metadata={"help": "name of backbone unet model in fairseq-signals"}
    )

class ECGUNetFinetuningModel(FinetuningModel):
    def __init__(self, cfg: ECGUNetFinetuningConfig, backbone: Union[ECGUNetModel, ECGUNet3PlusModel]):
        super().__init__(cfg, backbone)
        
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
    
    @classmethod
    def build_model(cls, cfg: ECGUNetFinetuningConfig, task: Task = None):
        """Build a new model instance."""
        backbone_cls = ECGUNetModel if cfg.backbone == "unet" else ECGUNet3PlusModel

        if cfg.model_path and not cfg.no_pretrained_weights:
            backbone = backbone_cls.from_pretrained(cfg.model_path, cfg)
        else:
            backbone = backbone_cls(cfg)
        
        return cls(cfg, backbone)
    
    def forward(self, source, **kwargs):
        args = {"source": source}
        ft = self.freeze_finetune_updates <= self.num_updates
        
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args, **kwargs)

        return res

class ConvBnRelu1d(nn.Module):
    def __init__(self, in_d, dim, kernel_size, dropout=0, conv_bias=False):
        super().__init__()
        assert kernel_size % 2 == 1, (
            "`kernel_size` should be a odd number to keep the output of conv to be the same length"
        )

        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_d, dim, kernel_size, stride=1, padding=padding, bias=conv_bias)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout1d(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_d, dim, kernel_size, dropout=0, conv_bias=False):
        super().__init__()
        self.conv1 = ConvBnRelu1d(in_d, dim, kernel_size, dropout, conv_bias)
        self.conv2 = ConvBnRelu1d(dim, dim, kernel_size, dropout, conv_bias)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_d, skip_dim, dim, kernel_size, dropout=0, conv_bias=False):
        super().__init__()
        padding = (kernel_size - 1) / 2 - 1
        self.up = nn.ConvTranspose1d(in_d, in_d, kernel_size-1, stride=2, padding=padding)
        self.conv1 = ConvBnRelu1d(in_d + skip_dim, dim, kernel_size, dropout, conv_bias)
        self.conv2 = ConvBnRelu1d(dim, dim, kernel_size, dropout, conv_bias)
    
    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2] != x.shape[2]:
            x = F.pad(x, (0, 1)) # pad last dimension of x by (0, 1)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpBlockForUNet3Plus(nn.Module):
    def __init__(self, all_dims, skip_dim, dim, kernel_size, dropout=0, conv_bias=False):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv1 = nn.Conv1d(all_dims[0], skip_dim, kernel_size, padding=padding, bias=conv_bias)
        self.conv2 = nn.Conv1d(all_dims[1], skip_dim, kernel_size, padding=padding, bias=conv_bias)
        self.conv3 = nn.Conv1d(all_dims[2], skip_dim, kernel_size, padding=padding, bias=conv_bias)
        self.conv4 = nn.Conv1d(all_dims[3], skip_dim, kernel_size, padding=padding, bias=conv_bias)
        self.conv5 = nn.Conv1d(all_dims[4], skip_dim, kernel_size, padding=padding, bias=conv_bias)
        self.aggregate = ConvBnRelu1d(skip_dim * 5, dim, kernel_size, dropout, conv_bias)
    
    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.aggregate(x)
        return x