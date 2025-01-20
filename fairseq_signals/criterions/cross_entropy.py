import math
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn.functional as F

from fairseq_signals import logging, metrics, meters
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round


@dataclass
class CrossEntropyCriterionConfig(Dataclass):
    weight: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "a manual rescaling weight given to each class. if given, has to be a Tensor "
                "of a size C and floating point dtype"
        }
    )
    focal_loss: bool = field(
        default=False,
        metadata={
            "help": "whether to apply focal loss"
        }
    )
    gamma: float = field(
        default=1.0,
        metadata={
            "help": "a value for gamma in focal loss"
        }
    )

@register_criterion("cross_entropy", dataclass = CrossEntropyCriterionConfig)
class CrossEntropyCriterion(BaseCriterion):
    def __init__(self, cfg: CrossEntropyCriterionConfig, task: Task):
        super().__init__(task)

        self.weight = cfg.weight
        self.focal_loss = cfg.focal_loss
        self.gamma = cfg.gamma
    
    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the logits and targets from the model
        """
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

        loss = F.cross_entropy(
            input=logits,
            target=target,
            weight=self.weight,
            reduction="none" if self.focal_loss or not reduce else "sum"
        )
        if self.focal_loss:
            y_pred = torch.exp(-loss)
            loss = (1 - y_pred) ** self.gamma * loss
            if reduce:
                loss = loss.sum()

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            sample_size = target.numel()
        return sample_size

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        """
        Get the logging output to display while training
        """
        logits = logits.reshape(-1, logits.size(-1)) # (B, ..., C) -> (N, C)
        target = target.reshape(-1) # (B, ...) -> (N, )

        preds = logits.argmax(dim=-1)

        count = target.numel()
        corr = (preds == target).sum().item()

        logging_output["correct"] = corr
        logging_output["count"] = count

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
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        if nsignals > 0:
            metrics.log_scalar(f"{prefix}nsignals", nsignals)
        
        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}correct"].sum / meters[f"_{prefix}total"].sum, 5
                )
                if meters[f"_{prefix}total"].sum > 0
                else float("nan")
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True