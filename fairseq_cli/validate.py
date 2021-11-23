#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain
import pprint
import omegaconf

import torch
from fairseq_signals import distributed_utils, tasks
from fairseq_signals.utils import checkpoint_utils, options, utils
from fairseq_signals.dataclass.utils import convert_namespace_to_omegaconf, overwrite_args_by_name
from fairseq_signals.logging import metrics, progress_bar
from fairseq_signals.utils.utils import reset_logging
from omegaconf import DictConfig, OmegaConf

from pytz import timezone
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")

def main(cfg: DictConfig, override_args = None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    
    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    
    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)
    
    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0
    
    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None
    
    # Load model
    logger.info(f"loading model from {cfg.common_eval.path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(cfg.common_eval.path, overrides)
    model_cfg = OmegaConf.merge(state["cfg"].model, eval(cfg.common_eval.model_overrides))

    task = tasks.setup_task(cfg.task)
    model = task.build_model(model_cfg)

    if state is not None:
        #XXX
        # for legacy wav2vec model
        # state["model"] = {k.replace('w2v_encoder.w2v_model.','encoder.'): v for k, v in state["model"].items()}

        # for legacy clocs model
        # state["model"] = {k.replace('clocs_encoder.clocs_model.encoder.w2v_model.','encoder.'): v for k, v in state["model"].items()}
        # state["model"].pop("encoder.mask_emb")

        model.load_state_dict(state["model"], strict = True)
        logger.info(f"Loaded pre-trained model from {cfg.common_eval.path}")
    else:
        raise FileNotFoundError(cfg.common_eval.path)
    
    # Move model to GPU
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()
    
    # Print args
    logger.info(pprint.pformat(dict(cfg)))

    # Build criterion
    criterion = task.build_criterion(cfg.criterion)
    criterion.eval()

    def _fp_convert_sample(sample):
        def apply_half(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.half)
            return t
            # return t.to(dtype = torch.half)
        
        def apply_float(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.float)
            return t
        
        if use_fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        else:
            sample = utils.apply_to_sample(apply_float, sample)
        
        return sample

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine = False, epoch = 1, task_cfg = cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset = dataset,
            max_tokens = cfg.dataset.max_tokens,
            max_signals = cfg.dataset.batch_size,
            # max_positions = ...,
            ignore_invalid_inputs = cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple = cfg.dataset.required_batch_size_multiple,
            seed = cfg.common.seed,
            num_shards = data_parallel_world_size,
            shard_id = data_parallel_rank,
            num_workers = cfg.dataset.num_workers,
            data_buffer_size = cfg.dataset.data_buffer_size
        ).next_epoch_itr(shuffle = False)
        progress = progress_bar.progress_bar(
            itr,
            log_format = cfg.common.log_format,
            log_interval = cfg.common.log_interval,
            prefix = f"valid on '{subset}' subset",
            default_log_format = ("tqdm" if not cfg.common.no_progress_bar else "simple")
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            with torch.no_grad():
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                sample = _fp_convert_sample(sample)
                _loss, _sample_size, log_output = task.valid_step(sample, model, criterion, subset)
                progress.log(log_output, step = i)
                log_outputs.append(log_output)
        
        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size = cfg.common.all_gather_list_size,
                group = distributed_utils.get_data_parallel_group()
            )
            log_outputs = list(chain.from_iterable(log_outputs))
        
        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()
        
        progress.print(log_output, tag = subset, step = i)

        if hasattr(task, "post_validate"):
            task.post_validate(model, log_output, agg, num_updates=0)

def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults = True
    )

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args = override_args
    )

if __name__ == "__main__":
    cli_main()