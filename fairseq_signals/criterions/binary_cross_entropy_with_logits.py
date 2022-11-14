import math
from collections import Counter
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
    threshold: float = field(
        default=0.5,
        metadata={"help": "threshold value for measuring accuracy"}
    )
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
    multi_class_multi_label: bool = field(
        default=False,
        metadata={
            "help": "whether to measure metrics based on multi-class & multi-label setting. "
                "if set, `sample` should have `is_multi_class` and `classes` as a key "
                "to indicate which samples are based on multi-class and which class indices are corresponded"
        }
    )
    per_log_keys: List[str] = field(
        default_factory = lambda: [],
        metadata={
            "help": "additionally log metrics for each of these keys, only applied for acc, auc"
        }
    )

@register_criterion(
    "binary_cross_entropy_with_logits", dataclass = BinaryCrossEntropyWithLogitsCriterionConfig
)
class BinaryCrossEntropyWithLogitsCriterion(BaseCriterion):
    def __init__(self, cfg: BinaryCrossEntropyWithLogitsCriterionConfig, task: Task):
        super().__init__(task)
        self.threshold = cfg.threshold
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

        self.multi_class_multi_label = cfg.multi_class_multi_label

        self.per_log_keys = cfg.per_log_keys

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

        for plk in self.per_log_keys:
            logging_output[plk + "_em_count"] = dict()
            logging_output[plk + "_em_correct"] = dict()
            if not self.training and self.report_auc:
                logging_output[plk + "_y_score"] = dict()
                logging_output[plk + "_y_true"] = dict()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            outputs = probs > self.threshold

            corr = 0
            count = 0
            em_count = 0
            em_corr = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            # true = torch.where(target == 1)
            # false = torch.where(target == 0)
            # tp = outputs[true].sum()
            # fn = outputs[true].numel() - tp
            # fp = outputs[false].sum()
            # tn = outputs[false].numel() - fp

            for prob, gt, classes, is_multi_class in zip(
                probs, target, sample["classes"], sample["is_multi_class"]
            ):
                prob = prob[classes]
                gt = gt[classes]

                em_count += 1
                if is_multi_class:
                    count += 1
                    max = prob.argmax()
                    if gt[max]:
                        corr += 1
                        em_corr += 1
                else:
                    count += len(classes)

                    output = (prob > self.threshold) == gt
                    corr += output.sum().item()
                    if output.all():
                        em_corr += 1

            logging_output["correct"] = corr
            logging_output["count"] = count
            logging_output["em_correct"] = em_corr
            logging_output["em_count"] = em_count

            # logging_output["tp"] = tp.item()
            # logging_output["fp"] = fp.item()
            # logging_output["tn"] = tn.item()
            # logging_output["fn"] = fn.item()

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
        
            for plk in self.per_log_keys:
                plk_ids = [log_id.item() for log_id in sample[plk]]
                for i, plk_id in enumerate(plk_ids):
                    #XXX temporary for logging per question_id for verify questions
                    if plk == "question_id" and sample["question_type"][i] != 0:
                        continue

                    classes = sample["classes"][i]
                    prob = probs[i][classes]
                    gt = target[i][classes]
                    is_multi_class = sample["is_multi_class"][i]

                    if plk_id in logging_output[plk + "_em_count"]:
                        logging_output[plk + "_em_count"][plk_id] += 1
                        if is_multi_class:
                            if gt[prob.argmax()]:
                                logging_output[plk + "_em_correct"][plk_id] += 1
                        else:
                            output = (prob > self.threshold) == gt
                            logging_output[plk + "_em_correct"][plk_id] += output.all().int().item()

                        if not self.training and self.report_auc:
                            logging_output[plk + "_y_score"][plk_id].append(probs[i].cpu().numpy())
                            logging_output[plk + "_y_true"][plk_id].append(target[i].cpu().numpy())
                    else:
                        logging_output[plk + "_em_count"][plk_id] = 1
                        if is_multi_class:
                            if gt[prob.argmax()]:
                                logging_output[plk + "_em_correct"][plk_id] = 1
                            else:
                                logging_output[plk + "_em_correct"][plk_id] = 0
                        else:
                            output = (prob > self.threshold) == gt
                            logging_output[plk + "_em_correct"][plk_id] = output.all().int().item()

                        if not self.training and self.report_auc:
                            logging_output[plk + "_y_score"][plk_id] = [probs[i].cpu().numpy()]
                            logging_output[plk + "_y_true"][plk_id] = [target[i].cpu().numpy()]

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
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log.get("_y_true", 0) for log in logging_outputs])
            y_score = np.concatenate([log.get("_y_score", 0) for log in logging_outputs])

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true)

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

        em_correct = sum(log.get("em_correct", 0) for log in logging_outputs)
        metrics.log_scalar("_em_correct", em_correct)

        em_total = sum(log.get("em_count", 0) for log in logging_outputs)
        metrics.log_scalar("_em_total", em_total)

        # tp = sum(log.get("tp", 0) for log in logging_outputs)
        # metrics.log_scalar("_tp", tp)
        # fp = sum(log.get("fp", 0) for log in logging_outputs)
        # metrics.log_scalar("_fp", fp)
        # tn = sum(log.get("tn", 0) for log in logging_outputs)
        # metrics.log_scalar("_tn", tn)
        # fn = sum(log.get("fn", 0) for log in logging_outputs)
        # metrics.log_scalar("_fn", fn)

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
                "em_accuracy",
                lambda meters: safe_round(
                    meters["_em_correct"].sum / meters["_em_total"].sum, 5
                )
                if meters["_em_total"].sum > 0
                else float("nan")
            )

            # metrics.log_derived(
            #     "precision",
            #     lambda meters: safe_round(
            #         meters["_tp"].sum / (meters["_tp"].sum + meters["_fp"].sum), 5
            #     )
            #     if (meters["_tp"].sum + meters["_fp"].sum) > 0
            #     else float("nan")
            # )

            # metrics.log_derived(
            #     "recall",
            #     lambda meters: safe_round(
            #         meters["_tp"].sum / (meters["_tp"].sum + meters["_fn"].sum), 5
            #     )
            #     if (meters["_tp"].sum + meters["_fn"].sum) > 0
            #     else float("nan")
            # )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsignals",
            "sample_size",
            "all_zeros",
            "all_zeros_t",
            "o_score",
            "c_score",
            "i_score",
            "correct",
            "count",
            "em_correct",
            "em_count",
            "tp",
            "fp",
            "tn",
            "fn",
            "_y_true",
            "_y_score"
        }

        for log_key in logging_outputs[0]:
            if log_key not in builtin_keys:
                if log_key.endswith("em_count"):
                    log_key = log_key.split("em_count")[0]
                    em_counts = [log[log_key + "em_count"] for log in logging_outputs]
                    em_corrects = [log[log_key + "em_correct"] for log in logging_outputs]
                    aggregated_em_counts = Counter()
                    aggregated_em_corrects = Counter()
                    for em_count, em_correct in zip(em_counts, em_corrects):
                        aggregated_em_counts.update(Counter(em_count))
                        aggregated_em_corrects.update(Counter(em_correct))
                    aggregated_em_counts = dict(aggregated_em_counts)
                    aggregated_em_corrects = dict(aggregated_em_corrects)

                    for log_id in aggregated_em_counts.keys():
                        key = log_key + str(log_id)
                        
                        metrics.log_scalar(
                            "_" + key + "_em_total",
                            aggregated_em_counts[log_id]
                        )
                        metrics.log_scalar(
                            "_" + key + "_em_correct",
                            aggregated_em_corrects[log_id]
                        )

                        if aggregated_em_counts[log_id] > 0:
                            key1 = "_" + key + "_em_correct"
                            key2 = "_" + key + "_em_total"
                            metrics.log_derived(
                                key + "_em_accuracy",
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