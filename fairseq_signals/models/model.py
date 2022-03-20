# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class for various fairseq_ecg models.
"""

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq_signals.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass
)
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)

def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(module.unwrapped_module, expected_type), \
            f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"

class BaseModel(nn.Module):
    """Base class for fairseq_signals models."""

    def __init__(self):
        super().__init__()
    
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            # do not set defaults so that settings defaults from various architectures still works
            gen_parser_from_dataclass(parser, dc(), delete_default = True)
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")
    
    def get_targets(self, sample, net_output):
        """get targets from either the sample or the net's output."""
        return sample["target"]
    
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        if torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim = -1)
            else:
                return F.softmax(logits, dim = -1)
        raise NotImplementedError
    
    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)
    
    def load_state_dict(
        self,
        state_dict,
        strict = True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in:class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model
        
        self.upgrade_state_dict(state_dict)
        
        return super().load_state_dict(state_dict, strict)
    
    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")
    
    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."
            
            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)