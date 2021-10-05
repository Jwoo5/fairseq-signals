import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from fairseq_signals import logging, metrics
from fairseq_signals.data.ecg import ecg_utils
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class BinaryCrossEntropyCriterionConfig(Dataclass):
    weight: Optional[List[float]] = field(
        default = None,
        metadata = {
            "help": "a manual rescaling weight given to the loss of each batch element."
            "if given, has to be a float list of size nbatch."
        }
    )
    report_auc: bool = field(
        default=False,
        metadata={"help": "whether to report auprc / auroc metric, used for valid step"}
    )

@register_criterion(
    "binary_cross_entropy", dataclass = BinaryCrossEntropyCriterionConfig
)
class BinaryCrossEntropyWithLogitsCriterion(BaseCriterion):
    def __init__(self, cfg: BinaryCrossEntropyCriterionConfig, task: Task):
        super().__init__(task)
        self.weight = cfg.weight
        self.report_auc = cfg.report_auc
    
    def forward(self, model, sample, reduce = True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, aggregate=True).float()
        probs = torch.sigmoid(logits)
        target = model.get_targets(sample, net_output)

        reduction = "none" if not reduce else "sum"

        loss = F.binary_cross_entropy(
            input = probs,
            target = target,
            weight = self.weight,
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
            outputs = (probs > 0.5)

            if probs.numel() == 0:
                corr = 0
                count = 0
                tp = 0
                tn = 0
                fp = 0
                fn = 0
            else:
                count = float(probs.numel())
                corr = (outputs == target).sum().item()

                true = torch.where(target == 1)
                false = torch.where(target == 0)
                tp = outputs[true].sum()
                fn = outputs[true].numel() - tp
                fp = outputs[false].sum()
                tn = outputs[false].numel() - fp

            logging_output["correct"] = corr
            logging_output["count"] = count

            logging_output["tp"] = tp.item()
            logging_output["fp"] = fp.item()
            logging_output["tn"] = tn.item()
            logging_output["fn"] = fn.item()

            if not self.training and self.report_auc:
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

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            valid_labels = np.where(y_true.sum(0) > 0)[0]
            y_true = y_true[:, valid_labels]
            y_score = y_score[:, valid_labels]

            auroc = roc_auc_score(y_true = y_true, y_score = y_score, average="micro")
            metrics.log_scalar(
                "auroc", auroc, round = 3
            )

            auprc = average_precision_score(y_true = y_true, y_score = y_score, average="micro")
            metrics.log_scalar(
                "auprc", auprc, round = 3
            )

        metrics.log_scalar("nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        tp = sum(log.get("tp", 0) for log in logging_outputs)
        metrics.log_scalar("_tp", tp)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        metrics.log_scalar("_fp", fp)
        tn = sum(log.get("tn", 0) for log in logging_outputs)
        metrics.log_scalar("_tn", tn)
        fn = sum(log.get("fn", 0) for log in logging_outputs)
        metrics.log_scalar("_fn", fn)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan")
            )

            metrics.log_derived(
                "precision",
                lambda meters: safe_round(
                    meters["_tp"].sum / (meters["_tp"].sum + meters["_fp"].sum), 5
                )
                if (meters["_tp"].sum + meters["_fp"].sum) > 0
                else float("nan")
            )

            metrics.log_derived(
                "recall",
                lambda meters: safe_round(
                    meters["_tp"].sum / (meters["_tp"].sum + meters["_fn"].sum), 5
                )
                if (meters["_tp"].sum + meters["_fn"].sum) > 0
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