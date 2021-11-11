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
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample
        
        Returns a tuple with three elements:
        1) the loss
        2) the sample_size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output)
        target = model.get_targets(sample, net_output)

        losses = []

        reduction = "none" if not reduce else "sum"

        loss = F.mse_loss(
            logits, target, reduction=reduction
        )

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        else:
            sample_size = target.numel()
        losses.append(loss.detach().clone())

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "ntokens": sample_size,
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

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False