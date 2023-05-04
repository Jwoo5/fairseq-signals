import contextlib
from dataclasses import dataclass, field
from typing import cast
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq_signals import tasks
from fairseq_signals.utils import utils
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_signals.models.pretraining_model import PretrainingConfig, PretrainingModel
from fairseq_signals.models.finetuning_model import FinetuningConfig, FinetuningModel
from fairseq_signals.tasks import Task


logger = logging.getLogger(__name__)

@dataclass
class VGG16Config(PretrainingConfig):
    """VGG-16 network configs (https://arxiv.org/abs/1409.1556)"""
    input_length: int = field(
        default=5000,
        metadata={
            "help": "size of input ecgs, which is used for the last linear layer"
        }
    )
    dropout: float = field(
        default=0.5, metadata={"help": "dropout probability"}
    )
    in_d: int = field(
        default=12, metadata={"help": "input dimension"}
    )

class VGG16Model(PretrainingModel):
    """VGG-16 model implementation (https://arxiv.org/abs/1409.1556)"""
    def __init__(self, cfg: VGG16Config):
        super().__init__(cfg)

        input_length = cfg.input_length
        in_channels = cfg.in_d
        dropout = cfg.dropout

        feature_extractor = []
        vgg_16_cfgs = [
            64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"
        ]
        for v in vgg_16_cfgs:
            if v == "M":
                feature_extractor.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                v = cast(int, v)
                feature_extractor += [
                    nn.Conv1d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v
        self.feature_extractor = nn.Sequential(*feature_extractor)
        
        output_length = self._get_feat_extract_output_lengths(vgg_16_cfgs)
        self.final_proj = nn.Sequential(
            nn.Linear(512 * output_length, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096)
        )
        
        self.num_updates = 0
    
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)
    
    def _get_feat_extract_output_lengths(self, cfgs: list):
        input_length = self.input_length
        for v in cfgs:
            if v == "M":
                input_length = input_length // 2
        
        return input_length
    
    def forward(self, source, normalize=False, **kwargs):
        x = self.feature_extractor(source)
        x = torch.flatten(x, 1)
        x = self.final_proj(x)

        print(x.shape)
        breakpoint()
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
        cfg: VGG16Config,
        **kwargs
    ):
        """
        Load a :class:`~fairseq_signals.models.VGG16Model` from a pre-trained model checkpoint.
        
        Args:
            model_path (str): a path to a pre-trained model state dict
            cfg (VGG16Config): cfg to override some arguments of pre-trained model
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
class VGG16FinetuningConfig(FinetuningConfig, VGG16Config):
    pass

class VGG16FinetuningModel(FinetuningModel):
    def __init__(self, cfg: VGG16FinetuningConfig, encoder: VGG16Model):
        super().__init__(cfg, encoder)
        
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
    
    @classmethod
    def build_model(cls, cfg: FinetuningConfig, task: Task):
        """Build a new model instance."""
        if cfg.model_path and not cfg.no_pretrained_weights:
            encoder = VGG16Model.from_pretrained(cfg.model_path, cfg)
        else:
            encoder = VGG16Model(cfg)
        
        return cls(cfg, encoder)

    def forward(self, source, **kwargs):
        args = {"source": source}
        
        ft = self.freeze_finetune_updates <= self.num_updates
        
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.encoder.extract_features(**args)
        
        return res