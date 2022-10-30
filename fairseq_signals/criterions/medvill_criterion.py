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
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample
        
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output)
        align_logits = logits["align_x"]
        mlm_logits = logits["mlm_x"]

        target = model.get_targets(sample, net_output)
        align_target = target["align_y"]
        mlm_target = target["mlm_y"]

        losses = []

        reduction = "none" if not reduce else "sum"

        loss = F.cross_entropy(mlm_logits, mlm_target, reduction=reduction)

        sample_size = mlm_target.numel()
        losses.append(loss.detach().clone())

        align_loss = F.binary_cross_entropy(
            align_logits,
            align_target,
            reduction=reduction
        ) * sample_size / align_target.numel()
        loss += align_loss
        losses.append(align_loss)

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "ntokens": sample_size,
            "nsignals": sample["id"].numel(),
            "sample_size": sample_size
        }

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()
        
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
        
        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsignals", nsignals)

        correct = sum(log.get("mlm_correct", 0) for log in logging_outputs)
        metrics.log_scalar("_mlm_correct", correct)

        total = sum(log.get("mlm_count", 0) for log in logging_outputs)
        metrics.log_scalar("_mlm_total", total)

        if total > 0:
            metrics.log_derived(
                "mlm_accuracy",
                lambda meters: safe_round(
                    meters["_mlm_correct"].sum / meters["_mlm_total"].sum, 5
                )
                if meters["_mlm_total"].sum > 0
                else float("nan")
            )

        correct = sum(log.get("align_correct", 0) for log in logging_outputs)
        metrics.log_scalar("_align_correct", correct)

        total = sum(log.get("align_count", 0) for log in logging_outputs)
        metrics.log_scalar("_align_total", total)

        if total > 0:
            metrics.log_derived(
                "align_accuracy",
                lambda meters: safe_round(
                    meters["_align_correct"].sum / meters["_align_total"].sum, 5
                )
                if meters["_align_total"].sum > 0
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
                        k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round=3)

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False