# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import warnings
from argparse import Namespace
from typing import Any, Callable, Dict, List

import torch

from fairseq_signals import metrics
from fairseq_signals.utils import utils
from fairseq_signals.data import BaseDataset, data_utils, iterators
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.dataclass.utils import gen_parser_from_dataclass

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class StatefulContainer(object):
    _state: Dict[str, Any] = dict()
    _factories: Dict[str, Callable[[], Any]] = dict()

    def add_factory(self, name, factory: Callable[[], Any]):
        self._factories[name] = factory
    
    def merge_state_dict(self, state_dict: Dict[str, Any]):
        self._state.update(state_dict)
    
    @property
    def state_dict(self) -> Dict[str, Any]:
        return self._state
    
    def __getattr__(self, name):
        if name not in self._state and name in self._factories:
            self._state[name] = self._factories[name]()

        if name in self._state:
            return self._state[name]
        
        raise AttributeError(f"Task state has no factory for attribute {name}")

class Task(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.

    Tasks have limited statefulness. In particular, state that needs to be
    saved to/loaded from checkpoints needs to be stored in the `self.state`
    :class:`StatefulContainer` object. For example::

        self.state_add_factory("dictionary", self.load_dictionary)
        print(self.state.dictionary)    # calls self.load_dictionary()
    
    This is necessary so that when loading checkpoints, we can properly
    recreate the task state after initializing the task instance.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    cfg: Dataclass
    datasets: Dict[str, BaseDataset]
    dataset_to_epoch_iter: Dict[BaseDataset, Any]
    state: StatefulContainer = None

    def __init__(self, cfg: Dataclass, **kwargs):
        self.cfg = cfg
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.state = StatefulContainer()
    
    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task.
        
        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        return cls(cfg, **kwargs)
    
    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.cfg, "data", "")

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        task_cfg: Dataclass = None,
        **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (Dataclass): optional task configuration stored in the checkpoint that can be used
                                    to load datasets
        """
        raise NotImplementedError
    
    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        
        Returns:
            a :class:`~fairseq_signals.data.BaseDataset` corresponding to *split*
        """
        from fairseq_signals.data import BaseDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], BaseDataset):
            raise TypeError("Datasets are expected to be of type BaseDataset")
        return self.datasets[split]
    
    def can_reuse_epoch_itr(self, dataset):
        # We can reuse the epoch iterator across epochs as long as the dataset
        # hasn't disabled it. We default to ``False`` here, although in practicer
        # this will be ``True`` for most datasets that inherit from
        # ``BaseDataset`` due to the base implementation there.
        return getattr(dataset, "can_reuse_epoch_itr_across_epochs", False)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens = None,
        max_signals = None,
        max_positions = None,
        ignore_invalid_inputs = False,
        required_batch_size_multiple = 1,
        seed = 1,
        num_shards = 1,
        shard_id = 0,
        num_workers = 0,
        epoch = 1,
        data_buffer_size = 0,
        disable_iterator_cache = False
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_signals (int, optional): max number of signals in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                signals that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]
        
        assert isinstance(dataset, BaseDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # # filter examples that are too large
        # if max_positions is not None:
        #     indices = self.filter_indices_by_size(
        #         indices, dataset, max_positions, ignore_invalid_inputs
        #     )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens = max_tokens,
            max_signals = max_signals,
            required_batch_size_multiple = required_batch_size_multiple
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset = dataset,
            collate_fn = dataset.collator,
            batch_sampler = batch_sampler,
            seed = seed,
            num_shards = num_shards,
            shard_id = shard_id,
            num_workers =num_workers,
            epoch = epoch,
            buffer_size = data_buffer_size
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter
        
        return epoch_iter

    def build_model(self, cfg: Dataclass):
        """
        Build the :class:`~fairseq_signals.BaseModel` instance for this
        task.

        Args:
            cfg (Dataclass): configuration object
        
        Returns:
            a :class:`~fairseq_signals.BaseModel` instance
        """
        from fairseq_signals import models

        model = models.build_model(cfg, self)
        return model
    
    def build_criterion(self, cfg: DictConfig):
        """
        Build the :class:`~fairseq_signals.criterions.BaseCriterion` instance for
        this task.

        Args:
            cfg (omegaconf.DictConfig): configuration object
        
        Returns:
            a :class:`~fairseq_signals.criterions.BaseCriterion` instance
        """
        from fairseq_signals import criterions

        return criterions.build_criterion(cfg, self)
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad = False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq_signals.data.BaseDataset`.
            model (~fairseq_signals.models.BaseModel): the model
            criterion(~fairseq_signals.criterions.BaseCriterion): the criterion
            optimizer (torch.optim.Optimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = torch.norm(param.grad.data, p=2, dtype=torch.float32)
        #         if torch.isnan(grad_norm).any() or torch.isinf(grad_norm).any():
        #             breakpoint()
        #             print()

        # for n, p in model.named_parameters():
        #     if p.grad is None and p.requires_grad is True:
        #         print("Parameter not used: ", n, p.shape)
        #     else:
        #         print("***Parameter used: ", n, p.shape)
        # breakpoint()

        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, criterion, subset=None):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
    
    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()
    
    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError
    
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass
    
    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # if not any("ntokens" in log for log in logging_outputs):
        #     warnings.warn(
        #         "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
        #     )
        # else:
        #     ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        #     metrics.log_scalar("wpb", ntokens, priority = 180, round = 1)
        #     metrics.log_speed("wps", ntokens, priority = 90, round = 1)
        
        criterion.__class__.reduce_metrics(logging_outputs)

    def state_dict(self):
        if self.state is not None:
            return self.state.state_dict
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.state is not None:
            self.state.merge_state_dict(state_dict)