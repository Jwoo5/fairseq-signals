import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq_signals import logging, metrics
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.logging.meters import safe_round

@dataclass
class MSECriterionConfig(Dataclass):
    pass

@register_criterion("mse", dataclass=MSECriterionConfig)
class MSECriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the logits and targets from the model
        """
        reduction = "none" if not reduce else "sum"

        loss = F.mse_loss(logits, target, reduction=reduction)

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        else:
            sample_size = target.numel()
        return sample_size

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

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False