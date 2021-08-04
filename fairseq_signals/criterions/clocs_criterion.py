from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

import numpy as np

from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass, ChoiceEnum
from fairseq_signals.tasks import Task
from fairseq_signals.logging.meters import safe_round

CLOCS_MODE_CHOICES = ChoiceEnum(["cmsc", "cmlc", "cmsmlc"])

@dataclass
class ClocsCriterionConfig(Dataclass):
    mode: CLOCS_MODE_CHOICES = field(
        default="cmsc", metadata={"help": "coding mode for clocs model"}
    )
    temp: float = field(
        default=0.1, metadata={"help": "temperature in softmax"}
    )
    eps: float = field(
        default=1e-8, metadata={"help": "small value for numerical stability"}
    )

@register_criterion("clocs", dataclass = ClocsCriterionConfig)
class ClocsCriterion(BaseCriterion):
    def __init__(self, cfg: ClocsCriterionConfig, task: Task):
        super().__init__(task)
        self.mode = cfg.mode
        self.temp = cfg.temp
        self.eps = cfg.eps
    
    def forward(self, model, sample, reduce = True):
        """Compute the loss for the given sample
        
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()

        losses = []
        loss = 0

        #TODO get_logits, get_targets

        if self.mode == "cmsc":
            logits = logits.transpose(0,1)
            logits /= torch.max(
                logits.detach().norm(dim=2).unsqueeze(2),
                self.eps * torch.ones_like(logits)
            )

            indices = torch.where(net_output["segment"] == 0)[0]
            mat1 = logits[:, indices, :]
            pat1 = net_output["patient_id"][indices.cpu()]

            indices = torch.where(net_output["segment"] == 1)[0]
            mat2 = logits[:, indices, :]
            pat2 = net_output["patient_id"][indices.cpu()]

            logits = torch.matmul(mat1, mat2.transpose(1,2))
            logits = logits / self.temp

            logits_1 = -F.log_softmax(logits, dim = -1)
            logits_2 = -F.log_softmax(logits.transpose(1,2), dim = -1)            

            target = torch.from_numpy(
                np.array([p == pat2 for p in pat1])
            ).to(logits.device)

            loss = torch.mean(
                torch.stack(
                    [torch.mean(l[target]) for l in logits_1]
                )
            )
            losses.append(loss.detach().clone())

            loss_2 = torch.mean(
                torch.stack(
                    [torch.mean(l[target]) for l in logits_2]
                )
            )
            loss += loss_2
            losses.append(loss_2.detach().clone())
        else:
            raise NotImplementedError()
        
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
            "loss", loss_sum / (len(logging_outputs) or 1 ), round = 3
        )

        metrics.log_scalar("nsignals", nsignals)
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False