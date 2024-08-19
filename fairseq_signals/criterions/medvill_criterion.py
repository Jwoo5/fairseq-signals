import math

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq_signals import logging, metrics
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round


@dataclass
class MedViLLCriterionConfig(Dataclass):
    pass

@register_criterion("medvill", dataclass=MedViLLCriterionConfig)
class MedViLLCriterion(BaseCriterion):
    def __init__(sefl, cfg: MedViLLCriterionConfig, task: Task):
        super().__init__(task)

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the logits and targets from the model
        """
        reduction = "none" if not reduce else "sum"

        align_logits = logits["align_x"]
        mlm_logits = logits["mlm_x"]

        align_target = target["align_y"]
        mlm_target = target["mlm_y"]

        losses = []

        loss = F.cross_entropy(mlm_logits, mlm_target, reduction=reduction)
        losses.append(loss.detach().item())

        align_loss = F.binary_cross_entropy(
            align_logits,
            align_target,
            reduction=reduction
        ) * mlm_target.numel() / align_target.numel()
        loss += align_loss
        losses.append(align_loss.detach().item())

        return loss, losses
    
    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        mlm_target = target["mlm_y"]
        sample_size = mlm_target.numel()
        return sample_size

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        """
        Get the logging output to display while training
        """
        align_logits = logits["align_x"]
        mlm_logits = logits["mlm_x"]

        align_target = target["align_y"]
        mlm_target = target["mlm_y"]

        with torch.no_grad():
            if mlm_logits.numel() == 0:
                mlm_corr = 0
                mlm_count = 0
            else:
                assert mlm_logits.dim() > 1, mlm_logits.shape
                mlm_corr = (mlm_logits.argmax(-1) == mlm_target).long().sum().item()
                mlm_count = float(mlm_target.numel())
            
            logging_output["mlm_correct"] = mlm_corr
            logging_output["mlm_count"] = mlm_count

            if align_logits.numel() == 0:
                align_corr = 0
                align_count = 0
            else:
                align_corr = ((align_logits > 0.5) == align_target).long().sum().item()
                align_count = float(align_target.numel())

            logging_output["align_correct"] = align_corr
            logging_output["align_count"] = align_count
        return logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs, prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        if prefix is None:
            prefix = ""
        elif prefix is not None and not prefix.endswith("_"):
            prefix = prefix + "_"

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )
        metrics.log_scalar(f"{prefix}ntokens", ntokens)
        metrics.log_scalar(f"{prefix}nsignals", nsignals)

        correct = sum(log.get("mlm_correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}mlm_correct", correct)

        total = sum(log.get("mlm_count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}mlm_total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}mlm_accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}mlm_correct"].sum / meters[f"_{prefix}mlm_total"].sum, 5
                )
                if meters[f"_{prefix}mlm_total"].sum > 0
                else float("nan")
            )

        correct = sum(log.get("align_correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}align_correct", correct)

        total = sum(log.get("align_count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}align_total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}align_accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}align_correct"].sum / meters[f"_{prefix}align_total"].sum, 5
                )
                if meters[f"_{prefix}align_total"].sum > 0
                else float("nan")
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsignals",
            "sample_size",
            "correct",
            "count"
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k,0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        prefix + k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(prefix + k, val / len(logging_outputs), round=3)

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False