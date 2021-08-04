# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch.utils.data
from fairseq_signals.data import data_utils

logger = logging.getLogger(__name__)

class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.

        This needs to return ``False`` if the sample size can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True
    
    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass

class BaseDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def collator(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError
    
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype = np.int64)
    
    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
    
    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)
    
    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        return NotImplementedError
    
    def batch_by_size(
        self,
        indices,
        max_tokens = None,
        max_signals = None,
        required_batch_size_multiple = 1
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_signals* and *required_batch_size_multiple*.
        """
        from fairseq_signals.data import data_utils

        try:
            num_tokens_vec = self.num_tokens_vec(indices).astype('int64')
        except NotImplementedError:
            num_tokens_vec = None
        
        return data_utils.batch_by_size(
            indices,
            num_tokens_fn = self.num_tokens,
            num_tokens_vec = num_tokens_vec,
            max_tokens = max_tokens,
            max_signals = max_signals,
            required_batch_size_multiple = required_batch_size_multiple
        )


    @property
    def support_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the works of the dataloader."""
        return True