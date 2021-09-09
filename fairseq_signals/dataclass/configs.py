# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

from omegaconf import II, MISSING

from fairseq_signals.dataclass.constants import (
    LOG_FORMAT_CHOICES,
    DDP_COMM_HOOK_CHOICES
)

@dataclass
class Dataclass:
    
    _name: Optional[str] = None

    @staticmethod
    def name():
        return None
    
    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)
    
    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name
    
    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith("${"):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif getattr(self, attribute_name) != self.__dataclass_fields__[attribute_name].default:
                return getattr(self, attribute_name)
        
        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default
    
    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type
    
    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")
    
    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")
    
    # def _get_argparse_alias(self, attribute_name: str) -> Any:
    #     return self._get_meta(attribute_name, "argparse_alias")
    
    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")

@dataclass
class CommonConfig(Dataclass):
    no_progress_bar: bool = field(
        default = False,
        metadata={"help": "disable progress bar"}
    )
    log_interval: int = field(
        default = 100,
        metadata={"help": "log progress every N batches"}
    )
    log_format: Optional[LOG_FORMAT_CHOICES] = field(
        default=None, metadata={"help": "log format to use"}
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "log file to copy metrics to."}
    )
    wandb_project: Optional[str] = field(
        default = None,
        metadata={"help": "Weights and Biases project name to use for logging"}
    )
    wandb_entity: Optional[str] = field(
        default = 'ecg-pretraining',
        metadata={"help": "Weights and Biases entity(team) name to use for logging"}
    )

    seed: int = field(
        default=1, metadata={"help": "pseudo random number generator seed"}
    )

    fp16: bool = field(default=False, metadata={"help": "use FP16"})
    memory_efficient_fp16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of FP16 training; implies --fp16"
        },
    )
    fp16_no_flatten_grads: bool = field(
        default=False, metadata={"help": "don't flatten FP16 grads tensor"}
    )
    fp16_init_scale: int = field(
        default=2 ** 7, metadata={"help": "default FP16 loss scale"}
    )
    fp16_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing loss scale"},
    )
    fp16_scale_tolerance: float = field(
        default=0.0,
        metadata={
            "help": "pct of updates that can overflow before decreasing the loss scale"
        },
    )
    on_cpu_convert_precision: bool = field(
        default=False,
        metadata={
            "help": "if set, the floating point conversion to fp16/bf16 runs on CPU. "
            "This reduces bus transfer time and GPU memory usage."
        }
    )
    min_loss_scale: float = field(
        default=1e-4,
        metadata={"help": "minimum FP16/AMP loss scale, after which training is stopped"},
    )
    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={"help": "threshold FP16 loss scale from below"}
    )

    empty_cache_freq: int = field(
        default=0,
        metadata={"help": "how often to clear the PyTorch CUDA cache (0 to disable)"},
    )
    all_gather_list_size: int = field(
        default=16384,
        metadata={"help": "number of bytes reserved for gathering stats from workers"},
    )
    model_parallel_size: int = field(
        default=1, metadata={"help": "total number of GPUs to parallelize model over"}
    )
    profile: bool = field(
        default=False, metadata={"help": "enable autograd profiler emit_nvtx"}
    )
    reset_logging: bool = field(
        default=False,
        metadata={
            "help": "when using Hydra, reset the logging at the beginning of training"
        },
    )
    suppress_crashes: bool = field(
        default=False,
        metadata={
            "help": "suppress crashes when training with the hydra_train entry point so that the "
                    "main method can return a value (useful for sweeps)"
        },
    )
@dataclass
class DistributedTrainingConfig(Dataclass):
    distributed_world_size: int = field(
        default = 1,
        metadata={"help": "total number of GPUs across all nodes"}
    )
    distributed_rank: Optional[int] = field(
        default = 0, metadata={"help": "rank of the current worker"}
    )
    disributed_backend: str = field(
        default = "nccl", metadata={"help": "distributed backend"}
    )
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "typically tcp://hostname:port that will be used to "
            "establish initial connetion"
        },
    )
    distributed_port: int = field(
        default = 12355,
        metadata={"help": "port number"}
    )
    device_id: int = field(
        default = 0,
        metadata={
            "help": "which GPU to use",
        }
    )
    ddp_comm_hook: DDP_COMM_HOOK_CHOICES = field(
        default="none", metadata={"help": "communication hook"}
    )
    bucket_cap_mb: int = field(
        default=25, metadata={"help": "bucket size for reduction"}
    )
    fix_batches_to_gpus: bool = field(
        default=False,
        metadata={
            "help": "don't shuffle batches between GPUs; this reduces overall "
            "randomness and may affect precision but avoids the cost of re-reading the data"
        },
    )
    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "disable unused parameter detection (not applicable to "
            "--ddp-backend=legacy_ddp)"
        },
    )
    heartbeat_timeout: int = field(
        default=-1,
        metadata={
            "help": "kill the job if no progress is made in N seconds; "
            "set to -1 to disable"
        },
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    fp16: bool = II("common.fp16")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
@dataclass
class DatasetConfig(Dataclass):
    num_workers: int = field(
        default = 1, metadata={"help": "how many subprocesses to use for data loading"}
    )
    # n_labels: Optional[int] = field(
    #     default = None, metadata = {"help": "the number of labels to be classified"}
    # )
    skip_invalid_size_inputs_valid_test: bool = field(
        default=False,
        metadata={"help": "ignore too long or too short lines in valid and test set"},
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "maximum number of tokens in a batch"}
    )
    batch_size: Optional[int] = field(
        default = None,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-signals"
        }
    )
    required_batch_size_multiple: int = field(
        default = 8, metadata = {"help": "batch size will be a multiplier of this value"}
    )
    data_buffer_size: int = field(
        default=10, metadata={"help": "Number of batches to preload"}
    )
    train_subset: str = field(
        default = "train",
        metadata={"help": "data subset to use for training (e.g. train, valid, test)"}
    )
    valid_subset: str = field(
        default="valid",
        metadata={"help": "comma separated list of data subsets to use for validation (e.g. train, valid, test)"}
    )
    combine_valid_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "comma separated list of data subsets to use for validation"
                    " (e.g. train, valid, test)",
            "argparse_alias": "--combine-val",
        },
    )
    ignore_unused_valid_subsets: Optional[bool] = field(
        default=False,
        metadata={"help": "do not raise error if valid subsets are ignored"},
    )
    validate_interval: int = field(
        default = 1, metadata={"help": "validate every N epochs"}
    )
    validate_interval_updates: int = field(
        default=0, metadata={"help": "validate every N updates"}
    )
    validate_after_updates: int = field(
        default=0, metadata={"help": "dont validate until reaching this many updates"}
    )
    fixed_validation_seed: Optional[int] = field(
        default=None, metadata={"help": "specified random seed for validation"}
    )
    disable_validation: bool = field(
        default = False, metadata={"help": "disable validation"}
    )
    max_tokens_valid: Optional[int] = field(
        default=II("dataset.max_tokens"),
        metadata={
            "help": "maximum number of tokens in a validation batch"
            " (defaults to --max-tokens)"
        },
    )
    batch_size_valid: Optional[int] = field(
        default = II("dataset.batch_size"), metadata={"help": "batch size of the validation batch (defaults to --batch-size)"}
    )
    max_valid_steps: Optional[int] = field(default=None, metadata={'help': 'How many batches to evaluate',
                                                                   "argparse_alias": "--nval"})
    curriculum: int = field(
        default=0, metadata={"help": "don't shuffle batches for first N epochs"}
    )
    num_shards: int = field(
        default=1, metadata={"help": "shard generation over N shards"}
    )
    shard_id: int = field(
        default=0, metadata={"help": "id of the shard to generate (id < num_shards)"}
    )

@dataclass
class OptimizationConfig(Dataclass):
    max_epoch: int = field(
        default = 0, metadata={"help": "force stop training at specified epoch"}
    )
    max_update: int = field(
        default=0, metadata={"help": "force stop training at specified update"}
    )
    lr: List[float] = field(
        default_factory=lambda: [0.25],
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    stop_time_hours: float = field(
        default=0,
        metadata={
            "help": "force stop training after specified cumulative time (if >0)"
        },
    )
    clip_norm: float = field(
        default=0.0, metadata={"help": "clip threshold of gradients"}
    )
    update_freq: List[int] = field(
        default_factory=lambda: [1],
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    stop_min_lr: float = field(
        default=-1.0,
        metadata={"help": "stop training when the learning rate reaches this minimum"},
    )

@dataclass
class CheckpointConfig(Dataclass):
    save_dir: str = field(
        default="checkpoints", metadata={"help": "path to save checkpoints"}
    )
    restore_file: str = field(
        default="checkpoint_last.pt",
        metadata={
            "help": "filename from which to load checkpoint "
            "(default: <save-dir>/checkpoint_last.pt"
        },
    )
    finetune_from_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "finetune from a pretrained model; note that meters and lr scheduler will be reset"
        },
    )
    reset_dataloader: bool = field(
        default=False,
        metadata={
            "help": "if set, does not reload dataloader state from the checkpoint"
        },
    )
    reset_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load lr scheduler state from the checkpoint"
        },
    )
    reset_meters: bool = field(
        default=False,
        metadata={"help": "if set, does not load meters from the checkpoint"},
    )
    reset_optimizer: bool = field(
        default=False,
        metadata={"help": "if set, does not load optimizer state from the checkpoint"},
    )
    optimizer_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override optimizer args when loading a checkpoint"
        },
    )
    save_interval: int = field(
        default=1, metadata={"help": "save a checkpoint every N epochs"}
    )
    save_interval_updates: int = field(
        default=0, metadata={"help": "save a checkpoint (and validate) every N updates"}
    )
    keep_interval_updates: int = field(
        default=-1,
        metadata={
            "help": "keep the last N checkpoints saved with --save-interval-updates"
        },
    )
    keep_interval_updates_pattern: int = field(
        default=-1,
        metadata={
            "help": "when used with --keep-interval-updates, skips deleting "
                    "any checkpoints with update X where "
                    "X %% keep_interval_updates_pattern == 0"
        },
    )
    keep_last_epochs: int = field(
        default=-1, metadata={"help": "keep last N epoch checkpoints"}
    )
    keep_best_checkpoints: int = field(
        default=-1, metadata={"help": "keep best N checkpoints based on scores"}
    )
    no_save: bool = field(
        default=False, metadata={"help": "don't save models or checkpoints"}
    )
    no_epoch_checkpoints: bool = field(
        default=False, metadata={"help": "only store last and best checkpoints"}
    )
    no_last_checkpoints: bool = field(
        default=False, metadata={"help": "don't store last checkpoints"}
    )
    no_save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "don't save optimizer-state as part of checkpoint"},
    )
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=False,
        metadata={
            "help": 'select the largest metric value for saving "best" checkpoints'
        },
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )
    checkpoint_suffix: str = field(
        default="", metadata={"help": "suffix to add to the checkpoint file name"}
    )
    checkpoint_shard_count: int = field(
        default=1,
        metadata={
            "help": "Number of shards containing the checkpoint - "
            "if the checkpoint is over 300GB, it is preferable "
            "to split it into shards to prevent OOM on CPU while loading "
            "the checkpoint"
        },
    )
    load_checkpoint_on_all_dp_ranks: bool = field(
        default=False,
        metadata={
            "help": "load checkpoints on all data parallel devices "
            "(default: only load on rank 0 and broadcast to other devices)"
        },
    )

@dataclass
class CommonEvalConfig(Dataclass):
    path: Optional[str] = field(
        default = None, metadata = {"help": "path to model file"}
    )
    quiet: bool = field(default = False, metadata = {"help": "only print final scores"})
    model_overrides: str = field(
        default = "{}",
        metadata = {
            "help": "a dictionary used to override model args at generation that were used during model training"
        }
    )
    results_path: Optional[str] = field(
        default = None,
        metadata = {"help": "path to save eval results (optional)"}
    )

@dataclass
class Config(Dataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    model: Any = MISSING
    task: Any = None
    criterion: Any = None
    lr_scheduler: Any = None
    optimizer: Any = None