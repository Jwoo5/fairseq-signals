from collections import Counter
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
from fairseq_signals.criterions import register_criterion
from fairseq_signals.criterions.binary_cross_entropy import (
    BinaryCrossEntropyCriterionConfig,
    BinaryCrossEntropyCriterion
)
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class MultiHeadBinaryCrossEntropyCriterionConfig(BinaryCrossEntropyCriterionConfig):
    log_per_class: bool = field(
        default=False,
        metadata={
            "help": "whether to also log metrics per class"
        }
    )
    per_log_keys: List[str] = field(
        default_factory = lambda: [],
        metadata={
            "help": "additionally log metrics for each of these keys, only applied for acc, auc"
        }
    )

@register_criterion(
    "multi_head_binary_cross_entropy", dataclass=MultiHeadBinaryCrossEntropyCriterionConfig
)
class MultiHeadBinaryCrossEntropyCriterion(BinaryCrossEntropyCriterion):
    def __init__(self, cfg: MultiHeadBinaryCrossEntropyCriterionConfig, task: Task):
        super().__init__(cfg, task)

        self.log_per_class = cfg.log_per_class
        self.per_log_keys = cfg.per_log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()

        target_idcs = sample["target_idx"]
        targets = sample["label"]

        logits = torch.cat([
            logits[i, target_idcs[i]] for i in range(len(logits))
        ])
        probs = torch.sigmoid(logits)
        targets = torch.cat(targets).float()

        # reduction = "none" if not reduce else "sum"
        # loss = F.binary_cross_entropy(
        #     input=probs,
        #     target=targets,
        #     weight=self.weight,
        #     reduction=reduction
        # )

        # if "sample_size" in sample:
        #     sample_size = sample["sample_size"]
        # else:
        #     sample_size = targets.numel()

        reduction = "none"
        losses = F.binary_cross_entropy(
            input=probs,
            target=targets,
            weight=self.weight,
            reduction=reduction
        )
        target_idcs = torch.cat(target_idcs).cpu().numpy()
        nums = Counter(target_idcs)
        loss = 0
        for l, t in zip(losses, target_idcs):
            loss += l / nums[t]
        loss /= len(nums)

        sample_size = 1

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "nsignals": sample["id"].numel(),
            "sample_size": sample_size
        }

        if self.log_per_class:
            logging_output["cls_count"] = dict()
            logging_output["cls_correct"] = dict()
            if not self.training and self.report_auc:
                logging_output["cls_y_score"] = dict()
                logging_output["cls_y_true"] = dict()

        for plk in self.per_log_keys:
            logging_output[plk + "_count"] = dict()
            logging_output[plk + "_correct"] = dict()
            if not self.training and self.report_auc:
                logging_output[plk + "_y_score"] = dict()
                logging_output[plk + "_y_true"] = dict()

        with torch.no_grad():
            outputs = probs > 0.5

            if probs.numel() == 0:
                count = 0
                corr = 0
            else:
                count = float(probs.numel())
                corr = (outputs == targets).sum().item()

            logging_output["correct"] = corr
            logging_output["count"] = count

            if not self.training and self.report_auc:
                logging_output["_y_true"] = targets.cpu().numpy()
                logging_output["_y_score"] = probs.cpu().numpy()

            if self.log_per_class:
                classes = torch.cat(target_idcs).cpu().numpy()
                for i, cls in enumerate(classes):
                    prob = probs[i]
                    gt = targets[i]
                    if cls not in logging_output["cls_count"]:
                        logging_output["cls_count"][cls] = 0
                        logging_output["cls_correct"][cls] = 0
                        if not self.training and self.report_auc:
                            logging_output["cls_y_score"][cls] = []
                            logging_output["cls_y_true"][cls] = []
                
                    logging_output["cls_count"][cls] += 1
                    output = (prob > 0.5) == gt
                    if output.item():
                        logging_output["cls_correct"][cls] += 1

                    if not self.training and self.report_auc:
                        logging_output["cls_y_score"][cls].append(prob.cpu().numpy())
                        logging_output["cls_y_true"][cls].append(gt.cpu().numpy())

                if not self.training and self.report_auc:
                    for cls in logging_output["cls_y_score"].keys():
                        logging_output["cls_y_score"][cls] = np.vstack(
                            logging_output["cls_y_score"][cls]
                        )
                        logging_output["cls_y_true"][cls] = np.vstack(
                            logging_output["cls_y_true"][cls]
                        )
            
            for plk in self.per_log_keys:
                plk_ids = np.concatenate(sample[plk])
                for i, plk_id in enumerate(plk_ids):
                    prob = probs[i]
                    gt = targets[i]
                    if plk_id not in logging_output[plk + "_count"]:
                        logging_output[plk + "_count"][plk_id] = 0
                        logging_output[plk + "_correct"][plk_id] = 0
                        if not self.training and self.report_auc:
                            logging_output[plk + "_y_score"][plk_id] = []
                            logging_output[plk + "_y_true"][plk_id] = []
                    
                    logging_output[plk + "_count"][plk_id] += 1
                    output = (prob > 0.5) == gt
                    if output.item():
                        logging_output[plk + "_correct"][plk_id] += 1

                    if not self.training and self.report_auc:
                        logging_output[plk + "_y_score"][plk_id].append(prob.cpu().numpy())
                        logging_output[plk + "_y_true"][plk_id].append(gt.cpu().numpy())

                if not self.training and self.report_auc:
                    for plk_id in logging_output[plk + "_y_score"].keys():
                        logging_output[plk + "_y_score"][plk_id] = np.vstack(
                            logging_output[plk + "_y_score"][plk_id]
                        )
                        logging_output[plk + "_y_true"][plk_id] = np.vstack(
                            logging_output[plk + "_y_true"][plk_id]
                        )

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
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("nsignals", nsignals)

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true)

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

        builtin_keys = {
            "loss",
            "nsignals",
            "sample_size",
            "correct",
            "count",
            "_y_true",
            "_y_score"
        }

        for log_key in logging_outputs[0]:
            if log_key not in builtin_keys:
                if log_key.endswith("count"):
                    log_key = log_key.split("count")[0]
                    counts = [log[log_key + "count"] for log in logging_outputs]
                    corrects = [log[log_key + "correct"] for log in logging_outputs]
                    aggregated_counts = Counter()
                    aggregated_corrects = Counter()
                    for count, correct in zip(counts, corrects):
                        aggregated_counts.update(Counter(count))
                        aggregated_corrects.update(Counter(correct))
                    aggregated_counts = dict(aggregated_counts)
                    aggregated_corrects = dict(aggregated_corrects)
                    
                    for log_id in aggregated_counts.keys():
                        key = log_key + str(log_id)

                        metrics.log_scalar(
                            "_" + key + "_total",
                            aggregated_counts[log_id]
                        )
                        metrics.log_scalar(
                            "_" + key + "_correct",
                            aggregated_corrects[log_id]
                        )

                        if aggregated_counts[log_id] > 0:
                            key1 = "_" + key + "_correct"
                            key2 = "_" + key + "_total"
                            metrics.log_derived(
                                key + "_accuracy",
                                lambda meters, key1=key1, key2=key2: safe_round(
                                    (
                                        meters[key1].sum
                                        / meters[key2].sum
                                    ), 5
                                )
                                if meters[key2].sum > 0
                                else float("nan")
                            )
                elif log_key.endswith("y_score"):
                    log_key = log_key.split("y_score")[0]
                    y_scores = [log[log_key + "y_score"] for log in logging_outputs]
                    y_trues = [log[log_key + "y_true"] for log in logging_outputs]

                    log_ids = set()
                    for vals in y_trues:
                        log_ids = log_ids.union(set(vals.keys()))

                    aggregated_scores = {log_id: [] for log_id in log_ids}
                    aggregated_trues = {log_id: [] for log_id in log_ids}
                    for y_score, y_true in zip(y_scores, y_trues):
                        for log_id in log_ids:
                            if log_id in y_score:
                                aggregated_scores[log_id].append(y_score[log_id])
                                aggregated_trues[log_id].append(y_true[log_id])

                    for log_id in log_ids:
                        aggregated_scores[log_id] = np.concatenate(aggregated_scores[log_id])
                        aggregated_trues[log_id] = np.concatenate(aggregated_trues[log_id])

                        key = log_key + str(log_id)

                        metrics.log_custom(
                            meters.AUCMeter,
                            "_" + key + "_auc",
                            aggregated_scores[log_id],
                            aggregated_trues[log_id]
                        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False