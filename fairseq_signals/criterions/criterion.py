# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Dict, List

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
        self.output_store = None
        self.target_store = None

    def set_output_store(self, output_store: Any):
        self.output_store = output_store

    def set_target_store(self, target_store: Any):
        self.target_store = target_store

    def store(self, output: Any, target: Any):
        if dist_utils.get_data_parallel_world_size() > 1:
            group = dist_utils.get_data_parallel_group()
            output = torch.cat(dist_utils.batch_all_gather(output, group=group))
            target = torch.cat(dist_utils.batch_all_gather(target, group=group))

        if self.output_store is not None:
            self.output_store(output)

        if self.target_store is not None:
            self.target_store(target)

    def close_stores(self):
        if self.output_store is not None:
            self.output_store.close()

        if self.target_store is not None:
            self.target_store.close()

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
    
    def forward(self, model, sample, reduce = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError
        
    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
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

