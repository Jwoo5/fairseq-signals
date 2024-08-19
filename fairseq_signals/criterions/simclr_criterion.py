from dataclasses import dataclass, field

import math

import torch
import torch.nn.functional as F

import numpy as np

from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.distributed import utils as dist_utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

@dataclass
class SimCLRCriterionConfig(Dataclass):
    temp: float = field(
        default=0.1, metadata={"help": "temperature in softmax"}
    )
    eps: float = field(
        default=1e-8, metadata={"help": "small value for numerical stability when normalizing"}
    )

@register_criterion("simclr", dataclass=SimCLRCriterionConfig)
class SimCLRCriterion(BaseCriterion):
    def __init__(self, cfg: SimCLRCriterionConfig, task: Task):
        super().__init__(task)
        self.temp = cfg.temp
        self.eps = cfg.eps

        self.kwargs["aggregate"] = True
        self.is_target_derived = True

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the logits and targets from the model
        """
        reduction = "none" if not reduce else "sum"

        logits /= torch.max(
            logits.detach().norm(dim=1).unsqueeze(1),
            self.eps * torch.ones_like(logits)
        )

        bsz = int(logits.shape[0] / 2)

        mask = 1 - torch.eye(bsz * 2, dtype=torch.uint8).to(logits.device)
        pos_ind = (
            torch.arange(bsz * 2).to(logits.device),
            2 * torch.arange(bsz, dtype=torch.long).unsqueeze(1).repeat(
                1, 2).view(-1, 1).squeeze().to(logits.device)
        )
        neg_mask = torch.ones((bsz * 2, bsz * 2 - 1), dtype=torch.uint8).to(logits.device)
        neg_mask[pos_ind] = 0
        
        sim_matrix = torch.matmul(logits, logits.T)

        sim_matrix = torch.masked_select(sim_matrix, mask.bool()).view(sim_matrix.size(0), -1)

        positives = sim_matrix[pos_ind].unsqueeze(1)
        negatives = torch.masked_select(sim_matrix, neg_mask.bool()).view(sim_matrix.size(0), -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temp

        target = torch.zeros((logits.size(0), ), dtype=torch.long).to(logits.device)

        loss = F.cross_entropy(logits, target, reduction=reduction)

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.numel()
        return sample_size

    def get_logging_output(self, logging_output, logits, target, sample=None, net_output=None):
        """
        Get the logging output to display while training
        """
        # TODO check if logits is calculated correctly after `compute_loss`
        # with torch.no_grad():
        #     if logits.numel() == 0:
        #         corr = 0
        #         count = 0
        #     else:
        #         assert logits.dim() > 1, logits.shape
        #         max = logits.argmax(-1) == 0
        #         min = logits.argmin(-1) == 0

        #         both = max & min
        #         corr = max.long().sum().item() - both.long().sum().item()
        #         count = float(max.numel())
            
        #     logging_output["correct"] = corr
        #     logging_output["count"] = count
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False