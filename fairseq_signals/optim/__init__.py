# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from fairseq_signals.utils import registry
from fairseq_signals.optim.optimizer import (
    Optimizer,
)
from fairseq_signals.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer

from omegaconf import DictConfig

__all__ = [
    "Optimizer",
    "FP16Optimizer",
    "MemoryEfficientFP16Optimizer"
]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY,
    OPTIMIZER_DATACLASS_REGISTRY
) = registry.setup_registry("--optimizer", base_class = Optimizer, required = True)

def build_optimizer(cfg: DictConfig, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    return _build_optimizer(cfg, params, *extra_args, **extra_kwargs)

# automatically import any Python files in the optim/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq_signals.optim." + file_name)