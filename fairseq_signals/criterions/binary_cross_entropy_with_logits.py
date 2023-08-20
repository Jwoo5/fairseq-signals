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
from fairseq_signals.criterions.binary_cross_entropy import BinaryCrossEntropyCriterionConfig, BinaryCrossEntropyCriterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class BinaryCrossEntropyWithLogitsCriterionConfig(BinaryCrossEntropyCriterionConfig):
    auc_average: str = field(
        default="macro",
        metadata={
            "help": "determines the type of averaging performed on the data, "
                "should be one of ['micro', 'macro']"
        }
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
    per_log_keys: List[str] = field(
        default_factory = lambda: [],
        metadata={
            "help": "additionally log metrics for each of these keys, only applied for acc, auc"
        }
    )

@register_criterion(
    "binary_cross_entropy_with_logits", dataclass = BinaryCrossEntropyWithLogitsCriterionConfig
)
class BinaryCrossEntropyWithLogitsCriterion(BinaryCrossEntropyCriterion):
    def __init__(self, cfg: BinaryCrossEntropyWithLogitsCriterionConfig, task: Task):
        super().__init__(cfg, task)
        self.auc_average = cfg.auc_average
        self.pos_weight = cfg.pos_weight
        self.report_cinc_score = cfg.report_cinc_score
        if self.report_cinc_score:
            assert cfg.weights_file
            classes, self.score_weights = (
                ecg_utils.get_physionet_weights(cfg.weights_file)
            )
            self.sinus_rhythm_index = ecg_utils.get_sinus_rhythm_index(classes)

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
            input=logits,
            target=target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=reduction
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

        per_log_keys = []
        for plk in self.per_log_keys:
            if plk in sample:
                per_log_keys.append(plk)
                logging_output[plk + "_em_count"] = dict()
                logging_output[plk + "_em_correct"] = dict()
                logging_output[plk + "_tp"] = dict()
                logging_output[plk + "_fp"] = dict()
                logging_output[plk + "_fn"] = dict()
                if not self.training and self.report_auc:
                    logging_output[plk + "_y_score"] = dict()
                    logging_output[plk + "_y_true"] = dict()
                    logging_output[plk + "_y_class"] = dict()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            outputs = probs > self.threshold

            if probs.numel() == 0:
                corr = 0
                count = 0
                em_count = 0
                em_corr = 0
                tp = 0
                fp = 0
                fn = 0
            else:
                if "valid_classes" in sample:
                    y_true = []
                    y_score = []
                    y_class = []
                    
                    for logit, prob, output, gt, classes in zip(
                        logits, probs, outputs, target, sample["valid_classes"]
                    ):
                        logit = logit[classes]
                        prob = prob[classes]
                        output = output[classes]
                        gt = gt[classes]

                        true = torch.where(gt == 1)
                        false = torch.where(gt == 0)
                        tp += output[true].sum()
                        fn += output[true].numel() - output[true].sum()
                        fp += output[false].sum()

                        em_count += 1
                        count += len(classes)

                        output = (output == gt)
                        corr += output.sum().item()
                        if output.all():
                            em_corr += 1

                        if not self.training and self.report_auc:
                            _y_true = gt.cpu().numpy()
                            _y_class = classes.cpu().numpy()
                            _y_score = prob.cpu().numpy()

                            y_true.append(_y_true)
                            y_score.append(_y_score)
                            if self.auc_average == "macro":
                                y_class.append(_y_class)
                    
                    if len(y_true) > 0:
                        y_true = np.concatenate(y_true)
                        y_score = np.concatenate(y_score)
                        if len(y_class) > 0:
                            y_class = np.concatenate(y_class)
                        else:
                            y_class = np.array([])
                else:
                    count = float(probs.numel())
                    corr = (outputs == target).sum().item()
                    em_count = probs.size(0)
                    em_corr = (outputs == target).all(axis=-1).sum().item()

                    true = torch.where(target == 1)
                    false = torch.where(target == 0)
                    tp = outputs[true].sum()
                    fn = outputs[true].numel() - tp
                    fp = outputs[false].sum()

                    if not self.training and self.report_auc:
                        y_true = target.cpu().numpy()
                        y_score = probs.cpu().numpy()

            logging_output["correct"] = corr
            logging_output["count"] = count
            logging_output["em_correct"] = em_corr
            logging_output["em_count"] = em_count
            if tp == 0 or fp == 0 or fn == 0:
                logging_output["tp"] = tp
                logging_output["fp"] = fp
                logging_output["fn"] = fn
            else:
                logging_output["tp"] = tp.item()
                logging_output["fp"] = fp.item()
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

            for plk in per_log_keys:
                plk_ids = [log_id.item() for log_id in sample[plk]]
                for i, plk_id in enumerate(plk_ids):
                    # NOTE -1 denotes for missing id that should be skipped in the aggregation
                    if plk_id == -1:
                        continue

                    if "valid_classes" in sample:
                        classes = sample["valid_classes"][i]
                        logit = logits[i][classes]
                        prob = probs[i][classes]
                        gt = target[i][classes]
                    else:
                        classes = np.arange(len(logits[i]))
                        logit = logits[i]
                        prob = probs[i]
                        gt = target[i]

                    output = (prob > self.threshold)

                    if plk_id not in logging_output[plk + "_em_count"]:
                        logging_output[plk + "_em_count"][plk_id] = 0
                        logging_output[plk + "_em_correct"][plk_id] = 0
                        logging_output[plk + "_tp"][plk_id] = 0
                        logging_output[plk + "_fp"][plk_id] = 0
                        logging_output[plk + "_fn"][plk_id] = 0
                        if not self.training and self.report_auc:
                            logging_output[plk + "_y_score"][plk_id] = []
                            logging_output[plk + "_y_true"][plk_id] = []
                            logging_output[plk + "_y_class"][plk_id] = []

                    true = torch.where(gt == 1)
                    false = torch.where(gt == 0)
                    logging_output[plk + "_tp"][plk_id] += output[true].sum().item()
                    logging_output[plk + "_fn"][plk_id] += output[true].numel() - output[true].sum().item()
                    logging_output[plk + "_fp"][plk_id] += output[false].sum().item()

                    logging_output[plk + "_em_count"][plk_id] += 1
                    output = (output == gt)
                    if output.all():
                        logging_output[plk + "_em_correct"][plk_id] += 1

                    if not self.training and self.report_auc:
                        _y_true = gt.cpu().numpy()
                        _y_score = prob.cpu().numpy()
                        
                        logging_output[plk + "_y_true"][plk_id].append(_y_true)
                        logging_output[plk + "_y_score"][plk_id].append(_y_score)
                        if self.auc_average == "macro":
                            _y_class = classes.cpu().numpy()
                            logging_output[plk + "_y_class"][plk_id].append(_y_class)

                if not self.training and self.report_auc:
                    for plk_id in logging_output[plk + "_y_score"].keys():
                        logging_output[plk + "_y_true"][plk_id] = np.concatenate(
                            logging_output[plk + "_y_true"][plk_id]
                        )
                        logging_output[plk + "_y_score"][plk_id] = np.concatenate(
                            logging_output[plk + "_y_score"][plk_id]
                        )
                        if self.auc_average == "macro":
                            logging_output[plk + "_y_class"][plk_id] = np.concatenate(
                                logging_output[plk + "_y_class"][plk_id]
                            )
                        else:
                            logging_output[plk + "_y_class"][plk_id] = np.array([])

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
            y_true = np.concatenate([log["_y_true"] for log in logging_outputs if "_y_true" in log])
            y_score = np.concatenate([log["_y_score"] for log in logging_outputs if "_y_score" in log])
            y_class = np.concatenate([log["_y_class"] for log in logging_outputs if "_y_class" in log])

            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true, y_class)

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

        tp = sum(log.get("tp", 0) for log in logging_outputs)
        metrics.log_scalar("_tp", tp)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        metrics.log_scalar("_fp", fp)
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
                "em_accuracy",
                lambda meters: safe_round(
                    meters["_em_correct"].sum / meters["_em_total"].sum, 5
                )
                if meters["_em_total"].sum > 0
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
                                        meters[key1].sum / meters[key2].sum
                                    ), 5
                                )
                                if meters[key2].sum > 0
                                else float("nan")
                            )

                # for precision / recall
                # elif log_key.endswith("tp"):
                #     log_key = log_key.split("tp")[0]
                #     tps = [log[log_key + "tp"] for log in logging_outputs]
                #     fps = [log[log_key + "fp"] for log in logging_outputs]
                #     fns = [log[log_key + "fn"] for log in logging_outputs]
                #     aggregated_tps = Counter()
                #     aggregated_fps = Counter()
                #     aggregated_fns = Counter()
                #     for tp, fp, fn in zip(tps, fps, fns):
                #         aggregated_tps.update(Counter(tp))
                #         aggregated_fps.update(Counter(fp))
                #         aggregated_fns.update(Counter(fn))
                #     aggregated_tps = dict(aggregated_tps)
                #     aggregated_fps = dict(aggregated_fps)
                #     aggregated_fns = dict(aggregated_fns)
                    
                #     for log_id in aggregated_tps.keys():
                #         key = log_key + str(log_id)
                        
                #         metrics.log_scalar(
                #             "_" + key + "_tp",
                #             aggregated_tps[log_id]
                #         )
                #         metrics.log_scalar(
                #             "_" + key + "_fp",
                #             aggregated_fps[log_id]
                #         )
                #         metrics.log_scalar(
                #             "_" + key + "_fn",
                #             aggregated_fns[log_id]
                #         )
                        
                #         key1 = "_" + key + "_tp"
                #         key2 = "_" + key + "_fp"
                #         key3 = "_" + key + "_fn"
                #         metrics.log_derived(
                #             key + "_precision",
                #             lambda meters, key1=key1, key2=key2: safe_round(
                #                 meters[key1].sum / (meters[key1].sum + meters[key2].sum), 5
                #             )
                #             if (meters[key1].sum + meters[key2].sum) > 0
                #             else float("nan")
                #         )
                #         metrics.log_derived(
                #             key + "_recall",
                #             lambda meters, key1=key1, key3=key3: safe_round(
                #                 meters[key1].sum / (meters[key1].sum + meters[key3].sum), 5
                #             )
                #             if (meters[key1].sum + meters[key3].sum) > 0
                #             else float("nan")
                #         )

                elif log_key.endswith("y_score"):
                    log_key = log_key.split("y_score")[0]
                    y_scores = [log[log_key + "y_score"] for log in logging_outputs]
                    y_trues = [log[log_key + "y_true"] for log in logging_outputs]
                    y_classes = [log[log_key + "y_class"] for log in logging_outputs]

                    log_ids = set()
                    for vals in y_trues:
                        log_ids = log_ids.union(set(vals.keys()))
                    
                    aggregated_scores = {log_id: [] for log_id in log_ids}
                    aggregated_trues = {log_id: [] for log_id in log_ids}
                    aggregated_classes = {log_id: [] for log_id in log_ids}
                    for y_score, y_true, y_class in zip(y_scores, y_trues, y_classes):
                        for log_id in log_ids:
                            if log_id in y_score:
                                aggregated_scores[log_id].append(y_score[log_id])
                                aggregated_trues[log_id].append(y_true[log_id])
                                aggregated_classes[log_id].append(y_class[log_id])

                    for log_id in log_ids:
                        aggregated_scores[log_id] = np.concatenate(aggregated_scores[log_id])
                        aggregated_trues[log_id] = np.concatenate(aggregated_trues[log_id])
                        aggregated_classes[log_id] = np.concatenate(aggregated_classes[log_id])

                        key = log_key + str(log_id)

                        metrics.log_custom(
                            meters.AUCMeter,
                            "_" + key + "_auc",
                            aggregated_scores[log_id],
                            aggregated_trues[log_id],
                            aggregated_classes[log_id]
                        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False