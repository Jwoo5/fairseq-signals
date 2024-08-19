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
class M3AECriterionConfig(Dataclass):
    norm_pix_loss: bool = field(
        default=True,
        metadata={"help": "whether to normalize ecg pixels channel-wisely before mim task"}
    )

@register_criterion("m3ae", dataclass=M3AECriterionConfig)
class M3AECriterion(BaseCriterion):
    def __init__(self, cfg: M3AECriterionConfig, task: Task):
        super().__init__(task)

        self.kwargs["norm_pix_loss"] = cfg.norm_pix_loss

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the final logits and targets directly fed to the loss function
        """
        reduction = "none" if not reduce else "mean"

        losses = []
        loss = F.cross_entropy(
            logits["mlm_logits"],
            target["mlm_target"],
            ignore_index=-100,
            reduction=reduction,
        )
        losses.append(loss.detach().item())

        mim_mask = net_output["mim_masks"]
        mim_loss = (logits["mim_logits"] - target["mim_target"]) ** 2
        mim_loss = mim_loss.mean(dim=-1)
        mim_loss = (mim_loss * mim_mask).sum() / mim_mask.sum()
        loss += mim_loss
        losses.append(mim_loss.detach().item())

        itm_loss = F.cross_entropy(
            logits["itm_logits"],
            target["itm_target"],
            reduction=reduction,
        )
        loss += itm_loss
        losses.append(itm_loss.detach().item())

        return loss, losses

    def get_sample_size(self, sample, targets):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        # set to 1 as the loss is already averaged in `compute_loss`
        return 1

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        """
        Get the logging output to display while training
        """
        with torch.no_grad():
            if logits["mlm_logits"].numel() == 0:
                mlm_corr = 0
                mlm_count = 0
            else:
                assert logits["mlm_logits"].dim() > 1, logits["mlm_logits"].shape
                mlm_corr = (logits["mlm_logits"].argmax(-1) == target["mlm_target"]).long().sum().item()
                mlm_count = (target['mlm_target'] != -100).sum()
                
                assert mlm_corr <= mlm_count
            logging_output["mlm_correct"] = mlm_corr
            logging_output["mlm_count"] = mlm_count
            
            if logits["itm_logits"].numel() == 0:
                itm_corr = 0
                itm_count = 0
            else:
                assert logits["itm_logits"].dim() > 1, logits["itm_logits"].shape
                itm_corr = (logits["itm_logits"].argmax(-1) == target["itm_target"]).long().sum().item()
                itm_count = target["itm_target"].numel()
            logging_output["itm_correct"] = itm_corr
            logging_output["itm_count"] = itm_count
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

        correct = sum(log.get("itm_correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}itm_correct", correct)

        total = sum(log.get("itm_count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}itm_total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}itm_accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}itm_correct"].sum / meters[f"_{prefix}itm_total"].sum, 5
                )
                if meters[f"_{prefix}itm_total"].sum > 0
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