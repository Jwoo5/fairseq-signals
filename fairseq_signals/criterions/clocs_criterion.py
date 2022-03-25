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
from fairseq_signals.models.clocs import CLOCS_MODE_CHOICES

@dataclass
class ClocsCriterionConfig(Dataclass):
    temp: float = field(
        default=0.1, metadata={"help": "temperature in softmax"}
    )
    eps: float = field(
        default=1e-8, metadata={"help": "small value for numerical stability when normalizing"}
    )
    clocs_mode: CLOCS_MODE_CHOICES = II("task.clocs_mode")

@register_criterion("clocs", dataclass = ClocsCriterionConfig)
class ClocsCriterion(BaseCriterion):
    def __init__(self, cfg: ClocsCriterionConfig, task: Task):
        super().__init__(task)
        self.mode = cfg.clocs_mode
        self.temp = cfg.temp
        self.eps = cfg.eps

    def forward(self, model, sample, reduce = True):
        """Compute the loss for the given sample
        
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert self.mode in ["cmsc", "cmlc", "cmsmlc"], self.mode

        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, aggregate=True).float()
        logits /= torch.max(
            logits.detach().norm(dim=-1).unsqueeze(-1),
            self.eps * torch.ones_like(logits)
        )

        patient_id = sample['patient_id']
        segment = sample['segment']

        losses = []
        loss = 0

        if self.mode == "cmsc":
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
        elif self.mode == "cmlc":
            combs = combinations(range(logits.size(0)), 2)
            logits = torch.stack(
                [torch.matmul(logits[first], logits[second].T) for first, second in combs]
            )
            logits /= self.temp

            target = torch.stack(
                [p == patient_id for p in patient_id]
            ).repeat(logits.size(0), 1, 1)
        else:
            indices = torch.where(segment == 0)[0]
            mat1 = logits[:, indices, :]
            p1 = (
                patient_id[indices]
            ) if len(indices) > 1 else (
                torch.tensor([patient_id[indices]])
            )

            indices = torch.where(segment == 1)[0]
            mat2 = logits[:, indices, :]
            p2 = (
                patient_id[indices]
            ) if len(indices) > 1 else (
                torch.tensor([patient_id[indices]])
            )

            combs = combinations(range(logits.size(0)), 2)
            logits = torch.stack(
                [torch.matmul(mat1[first], mat2[second].T) for first, second in combs]
            )
            logits /= self.temp

            target = torch.stack(
                ([p == p2 for p in p1])
            ).repeat(logits.size(0), 1, 1)

        logits_1 = -F.log_softmax(logits, dim = -1)
        logits_2 = -F.log_softmax(logits.transpose(-2,-1), dim = -1)

        loss_1 = logits_1[target].mean()
        loss += loss_1/2
        losses.append(loss_1.detach().clone())

        loss_2 = logits_2[target.transpose(-2, -1)].mean()
        loss += loss_2/2

        losses.append(loss_2.detach().clone())

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.long().sum().item()
        
        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "nsignals": sample["id"].numel(),
            "sample_size": sample_size
        }

        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        metrics.log_scalar("nsignals", nsignals)
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False