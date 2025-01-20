# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from fairseq_signals import logging, metrics
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.criterions.cmsc_criterion import CMSCCriterionConfig
from fairseq_signals.criterions.mse_criterion import MSECriterionConfig
from fairseq_signals.tasks import Task
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.logging.meters import safe_round

@dataclass
class Wav2Vec2CriterionConfig(Dataclass):
    infonce: bool = field(
        default = False,
        metadata = {
            "help": "if set, uses cross entropy instead of binary cross entropy (i.e. infoNCE loss)"
        }
    )
    loss_weights: Optional[List[float]] = field(
        default = None,
        metadata = {"help": "weights for additional loss terms (not first one)"}
    )
    log_keys: List[str] = field(
        default_factory = lambda: [],
        metadata = {"help": "output keys to log"}
    )

@register_criterion("wav2vec2", dataclass = Wav2Vec2CriterionConfig)
class Wav2Vec2Criterion(BaseCriterion):
    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the logits and targets from the model
        """
        reduction = "none" if not reduce else "sum"

        losses = []

        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )
        losses.append(loss.detach().item())

        sample_size = self.get_sample_size(sample, target)

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
                ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p.detach().item())

        return loss, losses

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.numel() if self.infonce else target.long().sum().item()
        return sample_size

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        """
        Get the logging output to display while training
        """
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing mAP and mAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    logging_output["target"] = target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                value = float(value)
                logging_output[lk] = value

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())
                
                logging_output["correct"] = corr
                logging_output["count"] = count
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
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )
        # metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar(f"{prefix}nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar(f"_{prefix}total", total)

        if total > 0:
            metrics.log_derived(
                f"{prefix}accuracy",
                lambda meters: safe_round(
                    meters[f"_{prefix}correct"].sum / meters[f"_{prefix}total"].sum, 5
                )
                if meters[f"_{prefix}total"].sum > 0
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
                        prefix + k, val / (sample_size or 1) / math.log(2), sample_size, round = 3
                    )
                else:
                    metrics.log_scalar(prefix + k, val / len(logging_outputs), round = 3)
    
    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

# TODO migrate "wav2vec2_with_cmsc" criterion to be constructed by "composite_criterion"

@dataclass
class Wav2Vec2WithCMSCCriterionConfig(Wav2Vec2CriterionConfig, CMSCCriterionConfig):
    cmsc_weights: Optional[float] = field(
        default=None,
        metadata={"help": "weights for cmsc loss terms"}
    )

@register_criterion("wav2vec2_with_cmsc", dataclass=Wav2Vec2WithCMSCCriterionConfig)
class Wav2Vec2WithCMSCCriterion(BaseCriterion):
    def __init__(self, cfg: Wav2Vec2WithCMSCCriterionConfig, task: Task):
        super().__init__(task)
        self.infonce = cfg.infonce
        self.loss_weights = cfg.loss_weights
        self.log_keys = [] if cfg.log_keys is None else cfg.log_keys
        self.temp = cfg.temp
        self.eps = cfg.eps
        self.cmsc_weights = cfg.cmsc_weights

    # custom forward function (cannot be used with composite criterion)
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        features = model.get_features(net_output, aggregate=True).float()
        w2v_target = model.get_targets(sample, net_output)

        self.store(logits, w2v_target.float(), net_output)

        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(w2v_target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = w2v_target.numel() if self.infonce else w2v_target.long().sum().item()

        losses = []

        reduction = "none" if not reduce else "sum"

        if self.infonce:
            loss = F.cross_entropy(logits, w2v_target, reduction = reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, w2v_target.float(), weights, reduction = reduction
            )
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
                ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)
        
        features /= torch.max(
            features.detach().norm(dim=-1).unsqueeze(-1),
            self.eps * torch.ones_like(features)
        )

        patient_id = sample["patient_id"]
        segment = sample["segment"]

        indices = torch.where(segment == 0)[0]
        mat1 = features[indices, :]
        p1 = (
            patient_id[indices]
        ) if len(indices) > 1 else (
            torch.tensor([patient_id[indices]])
        )

        indices = torch.where(segment == 1)[0]
        mat2 = features[indices, :]
        p2 = (
            patient_id[indices]
        ) if len(indices) > 1 else (
            torch.tensor([patient_id[indices]])
        )

        cmsc_logits = torch.matmul(mat1, mat2.transpose(0,1))
        cmsc_logits /= self.temp

        cmsc_target = torch.stack([p == p2 for p in p1]).to(cmsc_logits.device)
        
        logits_1 = -F.log_softmax(cmsc_logits, dim=-1)
        logits_2 = -F.log_softmax(cmsc_logits.transpose(-2,-1), dim=-1)

        loss_1 = logits_1[cmsc_target].mean()
        cmsc_loss = loss_1 / 2.0

        loss_2 = logits_2[cmsc_target.transpose(-2,-1)].mean()
        cmsc_loss += loss_2 / 2.0

        if self.cmsc_weights is not None:
            cmsc_loss = self.cmsc_weights * cmsc_loss * sample_size
        else:
            cmsc_loss *= sample_size
        loss += cmsc_loss
        losses.append(cmsc_loss.detach().clone())

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "nsignals": sample["id"].numel(),
            "sample_size": sample_size
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing mAP and mAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    logging_output["target"] = w2v_target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                value = float(value)
                logging_output[lk] = value

        for i, l in enumerate(losses):
            logging_output[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = float(max.numel())
                
                logging_output["correct"] = corr
                logging_output["count"] = count

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
                        k, val / (sample_size or 1) / math.log(2), sample_size, round = 3
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round = 3)
    
    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False