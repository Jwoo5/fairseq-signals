from dataclasses import dataclass, field
from omegaconf import II

import math
import torch

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
class ThreeKGCriterionConfig(Dataclass):
    temp: float = field(
        default=0.1, metadata={"help": "temperature in softmax"}
    )
    eps: float = field(
        default=1e-8, metadata={"help": "small value for numerical stability when normalizing"}
    )

    all_gather: bool = II("model.all_gather")

@register_criterion("3kg", dataclass=ThreeKGCriterionConfig)
class ThreeKGCriterion(BaseCriterion):
    def __init__(self, cfg: ThreeKGCriterionConfig, task: Task):
        super().__init__(task)
        self.temp = cfg.temp
        self.eps = cfg.eps
        self.all_gather = cfg.all_gather

        self.is_target_derived = True

    def compute_loss(
        self, logits, target=None, sample=None, net_output=None, model=None, reduce=True
    ):
        """
        Compute the loss given the final logits and targets directly fed to the loss function
        """
        logits /= torch.max(
            logits.detach().norm(dim=1).unsqueeze(1),
            self.eps * torch.ones_like(logits)
        )

        bsz = int(logits.shape[0] / 2)
        pids = sample['patient_id']

        # for all-gather tensor across distributed devices
        if self.all_gather and dist_utils.get_data_parallel_world_size() > 1:
            group = dist_utils.get_data_parallel_group()
            pids = torch.cat(
                dist_utils.batch_all_gather(pids, group=group)
            )

        sim_matrix = torch.matmul(logits, logits.T)        

        mask = 1 - torch.eye(bsz*2, dtype=torch.uint8).to(logits.device)
        sim_matrix = torch.masked_select(
            sim_matrix, mask==1
        ).view(sim_matrix.size(0), -1)

        pos_mask = torch.masked_select(
            torch.stack([p == pids for p in pids]), mask==1
        ).view(logits.size(0), -1)

        neg_mask = logits.new_ones((bsz*2, bsz*2-1), dtype=torch.uint8)
        neg_mask[pos_mask] = 0

        positives = sim_matrix[pos_mask].unsqueeze(1)

        negatives = sim_matrix.masked_fill_(pos_mask, -float('inf'))
        negatives = torch.cat(
            [torch.stack(
                [row] * torch.count_nonzero(row==-float('inf'))
            ) for row in negatives]
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temp

        target = torch.zeros((logits.size(0), ), dtype=torch.long).to(logits.device)

        reduction = "none" if not reduce else "sum"
        loss = F.cross_entropy(logits, target, reduction=reduction)

        return loss, [loss.detach().item()]

    def get_sample_size(self, sample, target=None):
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            sample_size = len(sample["net_input"]["source"])
        return sample_size

    def get_logging_output(
        self, logging_output, logits, target=None, sample=None, net_output=None
    ):
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
            f"{prefix}loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        if nsignals > 0:
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