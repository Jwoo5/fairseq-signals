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
from fairseq_signals.criterions.binary_cross_entropy import BinaryCrossEntropyCriterionConfig
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class BinaryCrossEntropyWithLogitsCriterionConfig(BinaryCrossEntropyCriterionConfig):
    pos_weight: Optional[List[float]] = field(
        default = None,
        metadata = {
            "help": "a weight of positive examples. Must be a vector with length equal to the"
            "number of classes."
        }
    )
    report_cinc_score: bool = field(
        default=False,
        metadata={"help": "whether to report cinc challenge metric"}
    )
    weights_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "score weights file for cinc challenge, only used when --report_cinc_score is True"
        }
    )

@register_criterion(
    "binary_cross_entropy_with_logits", dataclass = BinaryCrossEntropyWithLogitsCriterionConfig
)
class BinaryCrossEntropyWithLogitsCriterion(BaseCriterion):
    def __init__(self, cfg: BinaryCrossEntropyWithLogitsCriterionConfig, task: Task):
        super().__init__(task)
        self.weight = cfg.weight
        self.pos_weight = cfg.pos_weight
        self.report_auc = cfg.report_auc
        self.report_cinc_score = cfg.report_cinc_score
        if self.report_cinc_score:
            assert cfg.weights_file
            classes, self.score_weights = (
                ecg_utils.get_physionet_weights(cfg.weights_file)
            )
            self.sinus_rhythm_index = ecg_utils.get_sinus_rhythm_index(classes)
    
    def forward(self, model, sample, reduce = True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)

        reduction = "none" if not reduce else "sum"

        if self.pos_weight:
            self.pos_weight = torch.tensor(self.pos_weight).to(logits.device)

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

            if self.report_cinc_score:
                labels = target.cpu().numpy()

                observed_score = (
                    ecg_utils.compute_scored_confusion_matrix(
                        self.score_weights,
                        labels,
                        outputs.cpu().numpy()
                    )
                )
                correct_score = (
                    ecg_utils.compute_scored_confusion_matrix(
                        self.score_weights,
                        labels,
                        labels
                    )
                )
                inactive_outputs = np.zeros(outputs.size(), dtype=bool)
                inactive_outputs[:, self.sinus_rhythm_index] = 1
                inactive_score = (
                    ecg_utils.compute_scored_confusion_matrix(
                        self.score_weights,
                        labels,
                        inactive_outputs
                    )
                )

                logging_output["o_score"] = observed_score
                logging_output["c_score"] = correct_score
                logging_output["i_score"] = inactive_score

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

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true)

            if len(y_true) > 0:
                metrics.log_derived(
                    "auroc",
                    lambda meters: safe_round(
                        meters["_auc"].auroc, 3
                    )
                )
                metrics.log_derived(
                    "auprc",
                    lambda meters: safe_round(
                        meters["_auc"].auprc, 3
                    )
                )

        observed_score = sum(log.get("o_score", 0) for log in logging_outputs)
        metrics.log_scalar("_o_score", observed_score)

        correct_score = sum(log.get("c_score", 0) for log in logging_outputs)
        metrics.log_scalar("_c_score", correct_score)

        inactive_score = sum(log.get("i_score", 0) for log in logging_outputs)
        metrics.log_scalar("_i_score", inactive_score)

        if "o_score" in logging_outputs[0]:
            metrics.log_derived(
                "cinc_score",
                lambda meters: safe_round(
                    float(meters["_o_score"].sum - meters["_i_score"].sum) / (
                        float(meters["_c_score"].sum - meters["_i_score"].sum)
                    ), 3
                )
                if float(meters["_c_score"].sum - meters["_i_score"].sum) != 0
                else float(0.0)
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False