import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, Dict, Any
from omegaconf import MISSING

from fairseq_signals import logging, metrics, meters
from fairseq_signals.utils import utils
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks import Task

from . import build_criterion

@dataclass
class CompositeCriterionConfig(Dataclass):
    criterion_names: List[str] = field(
        default=MISSING,
        metadata={
            "help": "a list of underlying criterion names to be applied for each item in a list "
                "of model outputs and targets"
        }
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "weights for each loss term. if given, has to be a float list of size "
                "n_criterions"
        }
    )
    args: Any = field(
        default=None,
        metadata={
            "help": "configurations for each criterion where the name of each argument should "
                "match with the corresponding criterion. e.g., in case of "
                "`criterion_names=['cross_entropy', 'binary_cross_entropy']`, then configs for "
                "each criterion should be retrieved by `args.cross_entropy.*` and "
                "`args.binary_cross_entropy.*`"
        }
    )

@register_criterion("composite_criterion", dataclass=CompositeCriterionConfig)
class CompositeCriterion(BaseCriterion):
    """
    This is a composite criterion, that given a list of model logits and targets from
    `model.get_logits(...)` and `model.get_targets(...)`, computes losses for each
    logit-target pair.
    Note that, since it cannot expect which criterion and model will be used, it calls
    `model.get_logits(net_output, sample=sample)` without any other arguments. so, if you
    design a model to use this composite criterion, make sure that `get_logits(...)` method
    in the model works with only the two arguments `net_output` and `sample`.
    """

    def __init__(self, cfg: CompositeCriterionConfig, task: Task):
        super().__init__(task)

        self.criterion_names = cfg.criterion_names
        underlying_criterions = []
        for criterion_name in self.criterion_names:
            criterion_cfg = getattr(cfg.args, criterion_name)
            underlying_criterions.append(build_criterion(criterion_cfg, task))
        self.underlying_criterions = underlying_criterions

        if cfg.loss_weights is None:
            self.loss_weights = [1] * len(underlying_criterions)
        else:
            self.loss_weights = cfg.loss_weights

    def forward(self, model, sample, reduce=True, save_outputs=False):
        """
        Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model_kwargs = {}
        for criterion in self.underlying_criterions:
            model_kwargs.update(criterion.kwargs)

        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, sample=sample, **model_kwargs)
        try:
            targets = model.get_targets(sample, net_output, **model_kwargs)
        # some models / criterions don't need to implement get_targets as they derive targets from
        # logits (e.g., ThreeKGCriterion, etc)
        except NotImplementedError:
            targets = [None] * len(logits)

        if not isinstance(logits, list):
            logits = [logits]
        if not isinstance(targets, list):
            targets = [targets]

        if (
            len(logits) != len(self.underlying_criterions)
            or len(targets) != len(self.underlying_criterions)
        ):
            raise ValueError(
                "length of logits or targets do not match with the number of criterions. "
                f"Expected: {len(self.underlying_criterions)}, but got len(logits): {len(logits)}, "
                f"len(targets): {len(targets)}."
            )

        loss = 0
        logging_outputs = dict()
        for i, (logit, target) in enumerate(zip(logits, targets)):
            criterion = self.underlying_criterions[i]
            # if `forward` method is overriden by the underlying criterion
            if "forward" in criterion.__class__.__dict__:
                raise Exception(
                    "We don't support custom forward function for underlying criterions when "
                    f"using composite criterion, please check: {criterion.__class__}"
                )

            criterion.store(logit, target, net_output)

            partial_loss, partial_losses_to_log = self.loss_weights[i] * criterion.compute_loss(
                logits=logit,
                target=target,
                sample=sample,
                net_output=net_output,
                model=model,
                reduce=reduce
            )

            sample_size = criterion.get_sample_size(sample, target)

            logging_outputs[f"{i}_criterion_cls"] = self.underlying_criterions[i].__class__
            if len(partial_losses_to_log) > 1:
                logging_outputs[f"{i}_loss"] = (
                    partial_loss.item() if reduce else partial_loss.detach()
                )
                for j, l in enumerate(partial_losses_to_log):
                    logging_outputs[f"{i}_loss_{j}"] = l
            else:
                logging_outputs[f"{i}_loss"] = partial_losses_to_log[0]

            logging_outputs[f"{i}_sample_size"] = sample_size

            logging_output = criterion.get_logging_output({}, logit, target, sample, net_output)
            for log, value in logging_output.items():
                logging_outputs[f"{i}_{log}"] = value

            # divide partial loss by the partial sample size beforehand to handle different sample
            # sizes for multiple criterions
            loss += partial_loss / logging_outputs[f"{i}_sample_size"]

        # manipulate sample_size to be 1 to avoid dividing gradients in optimizer later
        sample_size = 1

        logging_outputs["nsignals"] = sample["id"].numel()

        return loss, sample_size, logging_outputs

    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        nsignals = utils.item(
            sum(log.get("nsignals", 0) for log in logging_outputs)
        )
        metrics.log_scalar("nsignals", nsignals)

        log_keys = logging_outputs[0].keys()

        grouped_log_keys = defaultdict(list)
        for lk in log_keys:
            group = lk.split("_")[-1]
            key = "_".join(lk.split("_")[:-1])
            if key != "" and group.isdigit():
                grouped_log_keys[group].append(key)
        grouped_log_keys = dict(sorted(grouped_log_keys.items()))
        grouped_log_keys = list(grouped_log_keys.values())

        total_loss = 0
        for i, log_keys in enumerate(grouped_log_keys):
            criterion_cls = logging_outputs[0][f"criterion_cls_{i}"]
            logging_output = []
            for log in logging_outputs:
                log_dict = {}
                for log_key in set(log_keys) - {"criterion_cls"}:
                    log_dict[log_key] = log[f"{log_key}_{i}"]
                logging_output.append(log_dict)
            criterion_cls.reduce_metrics(logging_output, prefix=str(i))

            loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_output))
            sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_output))

            total_loss += (loss_sum / (sample_size or 1) / math.log(2))

        metrics.log_scalar("loss", total_loss, 1, round=3)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

    def eval(self):
        super().eval()
        for criterion in self.underlying_criterions:
            criterion.eval()
        return self