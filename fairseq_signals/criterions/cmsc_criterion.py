from dataclasses import dataclass, field

import math

from typing import Optional
from omegaconf import II
from itertools import combinations

import torch
import torch.nn.functional as F

import numpy as np

from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.distributed import utils as dist_utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass, ChoiceEnum
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class CMSCCriterionConfig(Dataclass):
    temp: float = field(
        default=0.1, metadata={"help": "temperature in softmax"}
    )
    eps: float = field(
        default=1e-8, metadata={"help": "small value for numerical stability when normalizing"}
    )

    all_gather: bool = II("model.all_gather")

@register_criterion("cmsc", dataclass = CMSCCriterionConfig)
class CMSCCriterion(BaseCriterion):
    def __init__(self, cfg: CMSCCriterionConfig, task: Task):
        super().__init__(task)
        self.temp = cfg.temp
        self.eps = cfg.eps
        self.all_gather = cfg.all_gather

        self.is_target_derived = True

    def compute_loss(
        self, logits, target=None, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the final logits and targets directly fed to the loss function
        """
        logits /= torch.max(
            logits.detach().norm(dim=-1).unsqueeze(-1),
            self.eps * torch.ones_like(logits)
        )

        patient_id = sample['patient_id']
        segment = sample['segment']

        if self.all_gather and dist_utils.get_data_parallel_world_size() > 1:
            group = dist_utils.get_data_parallel_group()
            patient_id = torch.cat(
                dist_utils.batch_all_gather(patient_id, group=group)
            )
            segment = torch.cat(
                dist_utils.batch_all_gather(segment, group=group)
            )
        
        loss = 0

        indices = torch.where(segment == 0)[0]
        mat1 = logits[indices, :]
        p1 = (
            patient_id[indices]
        ) if len(indices) > 1 else (
            torch.tensor([patient_id[indices]])
        )

        indices = torch.where(segment == 1)[0]
        mat2 = logits[indices, :]
        p2 = (
            patient_id[indices]
        ) if len(indices) > 1 else (
            torch.tensor([patient_id[indices]])
        )

        logits = torch.matmul(mat1, mat2.transpose(0,1))
        logits /= self.temp
        target = torch.stack([p == p2 for p in p1]).to(logits.device)

        logits_1 = -F.log_softmax(logits, dim = -1)
        logits_2 = -F.log_softmax(logits.transpose(-2,-1), dim = -1)

        loss_1 = logits_1[target].mean()
        loss += loss_1/2

        loss_2 = logits_2[target.transpose(-2, -1)].mean()
        loss += loss_2/2

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.long().sum().item()

    def get_logging_output(
        self, logging_output, logits=None, target=None, sample=None, net_output=None
    ):
        """
        Get the logging output to display while training
        """
        return logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs, prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        if prefix is None:
            prefix = ""
        elif prefix is not None and not prefix.endswith("_"):
            prefix = prefix + "_"

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        metrics.log_scalar(f"{prefix}nsignals", nsignals)
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False