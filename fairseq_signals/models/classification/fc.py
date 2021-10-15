import contextlib

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from fairseq_signals.models import BaseModel, register_model
from fairseq_signals.models.conv_transformer import (
    MASKING_DISTRIBUTION_CHOICES,
    ConvTransformerModel,
    ConvTransformerConfig
)

from fairseq_signals.utils import utils
from fairseq_signals.tasks import Task