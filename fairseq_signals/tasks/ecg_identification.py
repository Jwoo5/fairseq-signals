from dataclasses import dataclass
import logging
import os
import sys

import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Optional, Any, List
from omegaconf import MISSING, II, OmegaConf

from fairseq_signals.distributed import utils as distributed_utils
from fairseq_signals.utils import utils
from fairseq_signals.data import IdentificationECGDataset
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks.ecg_pretraining import ECGPretrainingTask, ECGPretrainingConfig

from . import register_task

logger = logging.getLogger(__name__)

@dataclass
class ECGIdentificationConfig(ECGPretrainingConfig):
    num_labels: int=field(
        default=MISSING, metadata={"help": "number of patients to be classified when training"}
    )

    # The following are needed to load batch iterator for gallery sets
    #   when validating or testing
    max_tokens_valid: Optional[int] = II("dataset.max_tokens_valid")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    skip_invalid_size_inputs_valid_test: Optional[bool] = II("dataset.skip_invalid_size_inputs_valid_test")
    required_batch_size_multiple: Optional[int] = II("dataset.required_batch_size_multiple")
    seed: Optional[int] = II("common.seed")
    distributed_world_size: Optional[int] = II("distributed_training.distributed_world_size")
    num_workers: Optional[int] = II("dataset.num_workers")
    data_buffer_size: Optional[int] = II("dataset.data_buffer_size")
    fp16: Optional[bool] = II("common.fp16")

@register_task("ecg_identification", dataclass=ECGIdentificationConfig)
class ECGIdentificationTask(ECGPretrainingTask):
    cfg: ECGIdentificationConfig

    def __init__(
        self,
        cfg: ECGIdentificationConfig
    ):
        super().__init__(cfg)
        self.require_query = True
        self.gallery_feats = None
        self.gallery_pids = None
    
    @classmethod
    def setup_task(cls, cfg: ECGIdentificationConfig, **kwargs):
        """setup the task
        
        Args:
            cfg (ECGIdentificationConfig): configuration of this task
        """
        return cls(cfg)
    
    def load_dataset(self, split: str, task_cfg: Dataclass=None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg


        if 'train' in split:
            manifest_path = [os.path.join(data_path, "{}.tsv".format(split))]
            split = [split]
        else:
            if 'gallery' in split or 'probe' in split:
                logger.warning(
                    "'gallery' or 'probe' is included in split name, "
                    "which may not be intended. Please note that we infer "
                    "'gallery'/'probe' within the valid split names. "
                    "(e.g. when split == 'valid', we try to retrieve 'valid_gallery' "
                    "and 'valid_probe' on the fly.)"
                )
            manifest_path = [
                os.path.join(data_path, "{}_probe.tsv".format(split)),
                os.path.join(data_path, "{}_gallery.tsv".format(split))
            ]
            split = [f"{split}",f"{split}_gallery"]

        for i, s in enumerate(split):
            self.datasets[s] = IdentificationECGDataset(
                manifest_path=manifest_path[i],
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                label=True,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets,
                compute_mask_indices=self.cfg.precompute_mask_indices,
                **self._get_mask_precompute_kwargs(task_cfg)
            )

    def get_gallery_iterator(
        self,
        subset,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.get_batch_iterator(
            dataset=self.dataset(subset),
            max_tokens=self.cfg.max_tokens_valid,
            max_signals=self.cfg.batch_size_valid,
            ignore_invalid_inputs=self.cfg.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.required_batch_size_multiple,
            seed=self.cfg.seed,
            num_shards=1 if (
                self.cfg.distributed_world_size == 1 
            ) else distributed_utils.get_data_parallel_world_size(),
            shard_id=0 if (
                self.cfg.distributed_world_size == 1
            ) else distributed_utils.get_data_parallel_rank(),
            num_workers=self.cfg.num_workers,
            epoch=1,
            data_buffer_size=self.cfg.data_buffer_size,
            disable_iterator_cache=False
        )
        return batch_iterator

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype in [torch.float32, torch.int16]:
                return t.to(dtype = torch.half)
            return t
            # return t.to(dtype = torch.half)
        
        def apply_float(t):
            if t.dtype in [torch.float32, torch.int16]:
                return t.to(dtype = torch.float)
            return t

        if self.cfg.fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        else:
            sample = utils.apply_to_sample(apply_float, sample)
        
        return sample

    def _prepare_sample(self, sample, is_dummy=False):
        sample = utils.move_to_cuda(sample)
        sample = self._fp_convert_sample(sample)

        return sample, False

    def post_validate(self, model, log_output, agg):
        self.require_query = True
        self.gallery_feats = None
        self.gallery_pids = None

    def valid_step(self, sample, model, criterion, subset):
        model.eval()
        assert subset, subset
        with torch.no_grad():
            if self.require_query:
                itr = self.get_gallery_iterator(subset+"_gallery").next_epoch_itr(
                    shuffle=False, set_dataset_epoch=False
                )
                gallery_feats = []
                gallery_pids = []
                for gallery_sample in itr:
                    gallery_sample, _ = self._prepare_sample(gallery_sample)

                    net_output = model(**gallery_sample["net_input"])
                    feats = model.get_logits(net_output).float()
                    gallery_feats.append(feats)
                    gallery_pids.append(gallery_sample['patient_id'])

                self.gallery_feats = torch.cat(gallery_feats).unsqueeze(0)
                self.gallery_pids = np.concatenate(gallery_pids)
            self.require_query = False
            
            net_output = model(**sample["net_input"])
            probe_feats = model.get_logits(net_output).float()

            probe_feats = probe_feats.unsqueeze(2)
            cos_sims = torch.matmul(self.gallery_feats, probe_feats).squeeze(-1)
            select = torch.argmax(cos_sims, dim=1).cpu().numpy()
            best_pids = self.gallery_pids[select]
            
            outputs = (best_pids == sample['patient_id'])

            count = outputs.size
            corr = outputs.sum()

            if 'sample_size' in sample:
                sample_size = sample['sample_size']
            else:
                sample_size = outputs.size

            logging_output = {
                "loss": 0,
                "nsignals": sample["id"].numel(),
                "sample_size": sample_size,
                "correct": corr,
                "count": count
            }

            return 0, sample_size, logging_output