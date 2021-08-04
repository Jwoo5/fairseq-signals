import math
from argparse import Namespace
from dataclasses import dataclass, field
from numpy import average
from omegaconf import II
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
# from fairseq_signals.data.data_utils import post_process
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

#TODO develop how to calculate auprc, auroc, ...

@dataclass
class BinaryCrossEntropyCriterionConfig(Dataclass):
    weight: Optional[List[float]] = field(
        default = None,
        metadata = {
            "help": "a manual rescaling weight given to the loss of each batch element."
            "if given, has to be a float list of size nbatch."
        }
    )
    pos_weight: Optional[List[float]] = field(
        default = None,
        metadata = {
            "help": "a weight of positive examples. Must be a vector with length equal to the"
            "number of classes."
        }
    )
    report_auc: bool = field(
        default = False,
        metadata = {"help": "whether to report auprc / auroc metric, used for valid step"}
    )

@register_criterion(
    "binary_cross_entropy_with_logits", dataclass = BinaryCrossEntropyCriterionConfig
)
class BinaryCrossEntropyWithLogitsCriterion(BaseCriterion):
    def __init__(self, cfg: BinaryCrossEntropyCriterionConfig, task: Task):
        super().__init__(task)
        self.weight = cfg.weight
        self.pos_weight = cfg.pos_weight
        self.report_auc = cfg.report_auc
    
    def forward(self, model, sample, reduce = True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, aggregate = True).float()
        target = model.get_targets(sample, net_output)

        reduction = "none" if not reduce else "sum"

        loss = F.binary_cross_entropy_with_logits(
            input = logits,
            target = target,
            weight = self.weight,
            pos_weight = self.pos_weight,
            reduction = reduction
        )

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

        # for lk in self.log_keys:
        #     # Only store "logits" and "target" for computing acc, AUPRC, and AUROC
        #     # during validation
        #     if lk == "logits":
        #         if not self.training:
        #             logging_output["logits"] = logits.cpu().numpy()
        #     elif lk == "target":
        #         if not self.training:
        #             logging_output["target"] = target.cpu().numpy()
        #     elif lk in net_output:
        #         value = net_output[lk]
        #         value = float(value)
        #         logging_output[lk] = value

        with torch.no_grad():
            probs = torch.sigmoid(logits)

            if probs.numel() == 0:
                corr = 0
                count = 0
            else:
                count = float(probs.numel())
                corr = ((probs > 0.5) == target).sum().item()

            logging_output["correct"] = corr
            logging_output["count"] = count

            if self.report_auc:
                logging_output["_y_true"] = target.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()
        
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

        y_true = [log.get("_y_true", None) for log in logging_outputs]
        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            auroc = roc_auc_score(y_true = y_true, y_score = y_score)
            metrics.log_scalar(
                "auroc", auroc, round = 3
            )

            auprc = average_precision_score(y_true = y_true, y_score = y_score)
            metrics.log_scalar(
                "auprc", auprc, round = 3
            )

        metrics.log_scalar("nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan")
            )
        
        # builtin_keys = {
        #     "loss",
        #     "ntokens",
        #     "nsignals",
        #     "sample_size",
        #     "correct",
        #     "count"
        # }

        # for k in logging_outputs[0]:
        #     if k not in builtin_keys:
        #         val = sum(log.get(k,0) for log in logging_outputs)
        #         if k.startswith("loss"):
        #             metrics.log_scalar(
        #                 k, val / (sample_size or 1) / math.log(2), sample_size, round = 3
        #             )
        #         else:
        #             metrics.log_scalar(k, val / len(logging_outputs), round = 3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False