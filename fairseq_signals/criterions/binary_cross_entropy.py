import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from fairseq_signals import logging, metrics, meters
from fairseq_signals.data.ecg import ecg_utils
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class BinaryCrossEntropyCriterionConfig(Dataclass):
    weight: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "a manual rescaling weight given to the loss of each batch element."
            "if given, has to be a float list of size nbatch."
        }
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "threshold value for measuring accuracy"}
    )
    report_auc: bool = field(
        default=False,
        metadata={"help": "whether to report auprc / auroc metric, used for valid step"}
    )

@register_criterion(
    "binary_cross_entropy", dataclass = BinaryCrossEntropyCriterionConfig
)
class BinaryCrossEntropyCriterion(BaseCriterion):
    def __init__(self, cfg: BinaryCrossEntropyCriterionConfig, task: Task):
        super().__init__(task)
        self.threshold = cfg.threshold
        self.weight = cfg.weight
        self.report_auc = cfg.report_auc

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(
            input=probs,
            target=target,
            weight=self.weight,
            reduction="none" if not reduce else "sum"
        )
        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.long().sum().item()
        return sample_size

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            output = (probs > 0.5)

            if probs.numel() == 0:
                corr = 0
                count = 0
                tp = 0
                tn = 0
                fp = 0
                fn = 0
            else:
                count = float(probs.numel())
                corr = (output == target).sum().item()

                true = torch.where(target == 1)
                false = torch.where(target == 0)
                tp = output[true].sum()
                fn = output[true].numel() - tp
                fp = output[false].sum()
                tn = output[false].numel() - fp

            logging_output["correct"] = corr
            logging_output["count"] = count

            logging_output["tp"] = tp.item()
            logging_output["fp"] = fp.item()
            logging_output["tn"] = tn.item()
            logging_output["fn"] = fn.item()

            if not self.training and self.report_auc:
                logging_output["_y_true"] = target.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

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

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, f"_{prefix}auc", y_score, y_true)

        if nsignals > 0:
            metrics.log_scalar(f"{prefix}nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}total", total)

        tp = sum(log.get("tp", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}tp", tp)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}fp", fp)
        fn = sum(log.get("fn", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}fn", fn)

        if total > 0:
            metrics.log_derived(
                f"{prefix}accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}correct"].sum / meters[f"_{prefix}total"].sum, 5
                )
                if meters[f"_{prefix}total"].sum > 0
                else float("nan")
            )

            metrics.log_derived(
                f"{prefix}recall",
                lambda meters: safe_round(
                    meters[f"_{prefix}tp"].sum / (meters[f"_{prefix}tp"].sum + meters[f"_{prefix}fn"].sum), 5
                )
                if (meters[f"_{prefix}tp"].sum + meters[f"_{prefix}fn"].sum) > 0
                else float("nan")
            )

            metrics.log_derived(
                f"{prefix}recall",
                lambda meters: safe_round(
                    meters[f"_{prefix}tp"].sum / (meters[f"_{prefix}tp"].sum + meters[f"_{prefix}fn"].sum), 5
                )
                if (meters[f"_{prefix}tp"].sum + meters[f"_{prefix}fn"].sum) > 0
                else float("nan")
            )
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False