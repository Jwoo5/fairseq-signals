# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__ #noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

from fairseq_signals.distributed import utils as distributed_utils
from fairseq_signals.logging import meters, metrics

sys.modules["fairseq_signals.distributed_utils"] = distributed_utils
sys.modules["fairseq_signals.meters"] = meters
sys.modules["fairseq_signals.metrics"] = metrics

# initialize hydra
from fairseq_signals.dataclass.initialize import hydra_init
hydra_init()

import fairseq_signals.criterions # noqa
import fairseq_signals.distributed # noqa
import fairseq_signals.models # noqa
import fairseq_signals.modules # noqa
import fairseq_signals.optim
import fairseq_signals.optim.lr_scheduler
from fairseq_signals.utils import pdb
import fairseq_signals.tasks # noqa
