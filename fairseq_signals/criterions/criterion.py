# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Dict, List, Tuple

import torch

import fairseq_signals.distributed.utils as dist_utils
from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.dataclass.utils import gen_parser_from_dataclass
from torch.nn.modules.loss import _Loss

class BaseCriterion(_Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task

        self.stores = {}

        self.kwargs = {}
        self.is_target_derived = False

    def set_store(self, store_key: str, store: Any):
        self.stores[store_key] = store

    def store(self, output: Any, target: Any, net_output: Any):
        if len(self.stores) == 0:
            return # do nothing

        if dist_utils.get_data_parallel_world_size() > 1:
            group = dist_utils.get_data_parallel_group()
            # TODO handle when output and net_output are not torch tensors; it usually is python dict
            output = torch.cat(dist_utils.batch_all_gather(output, group=group))
            net_output = torch.cat(dist_utils.batch_all_gather(net_output, group=group))

            # some models & criterions do not yield targets
            if target is not None:
                target = torch.cat(dist_utils.batch_all_gather(target, group=group))

        if 'output' in self.stores:
            self.stores['output'](output)

        if 'target' in self.stores:
            self.stores['target'](target)

        if 'encoder_out' in self.stores:
            self.stores['encoder_out'](net_output['encoder_out'])

        if 'padding_mask' in self.stores:
            self.stores['padding_mask'](net_output['padding_mask'])

        if 'saliency' in self.stores:
            self.stores['saliency'](net_output['saliency'])

    def close_stores(self):
        for store in self.stores.values():
            store.close()

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_criterion(cls, cfg: Dataclass, task):
        """Construct a crtierion from command-line args."""
        # arguments in the __init__.
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (
                p.kind == p.POSITIONAL_ONLY
                or p.kind == p.VAR_POSITIONAL
                or p.kind == p.VAR_KEYWORD
            ):
                raise NotImplementedError("{} not supported".format(p.kind))
            
            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == "task":
                init_args["task"] = task
            elif p.name == "cfg":
                init_args["cfg"] = cfg
            elif hasattr(cfg, p.name):
                init_args[p.name] = getattr(cfg, p.name)
            elif p.default != p.empty:
                pass # we'll use the default value
            else:
                raise NotImplementedError(
                    "Unable to infer Criterion arguments, please implement "
                    "{}.build_criterion".format(cls.__name__)
                )
        return cls(**init_args)

    def compute_loss(
        self, logits, target, sample=None, net_output=None, model=None, reduce=True
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute the loss given the logits and targets from the model
        """
        raise NotImplementedError("Criterion must implement the `compute_loss` method")

    def get_sample_size(self, sample, target) -> int:
        """
        Get the sample size, which is used as the denominator for the gradient
        """
        raise NotImplementedError("Crietrion must implement the `get_sample_size` mtehod")

    def get_logging_output(
        self, logging_output, logits, target, sample=None
    ) -> List[Dict[str, Any]]:
        """
        Get the logging output to display while training
        """
        raise NotImplementedError("Criterion must implement the `log` method")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output, sample=sample, **self.kwargs)
        # some models / criterions don't need to implement get_targets as they derive targets from
        # logits (e.g., ThreeKGCriterion, etc)
        if not self.is_target_derived:
            targets = model.get_targets(sample, net_output, **self.kwargs)
        else:
            targets = None

        self.store(logits, targets, net_output)

        # TODO check logits before / after self.compute_loss(...)
        loss, losses_to_log = self.compute_loss(
            logits, targets, sample=sample, net_output=net_output, model=model, reduce=reduce
        )
        sample_size = self.get_sample_size(sample, targets)

        logging_output = {}
        if len(losses_to_log) > 1:
            logging_output["loss"] = loss.item() if reduce else loss.detach()
            for i, l in enumerate(losses_to_log):
                logging_output[f"loss_{i}"] = l
        else:
            logging_output["loss"] = losses_to_log[0]
        logging_output["nsignals"] = sample["id"].numel()
        logging_output["sample_size"] = sample_size
        logging_output = self.get_logging_output(
            logging_output, logits, targets, sample, net_output
        )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]], prefix: str = None) -> None:
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed taining speed.
        """
        return False

