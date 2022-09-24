import logging
import os

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional

from fairseq_signals.data import FileECGDataset
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks.ecg_pretraining import ECGPretrainingTask, ECGPretrainingConfig

from . import register_task

logger = logging.getLogger(__name__)

@dataclass
class ECGDiagnosisConfig(ECGPretrainingConfig):
    pass

@register_task("ecg_diagnosis", dataclass=ECGDiagnosisConfig)
class ECGDiagnosisTask(ECGPretrainingTask):
    cfg: ECGDiagnosisConfig

    def __init__(
        self,
        cfg: ECGDiagnosisConfig
    ):
        super().__init__(cfg)
    
    @classmethod
    def setup_task(cls, cfg: ECGDiagnosisConfig, **kwargs):
        """Setup the task
        
        Args:
            cfg (ECGDiagnosisConfig): configuration of this task
        """
        return cls(cfg)
    
    def load_dataset(self, split: str, task_cfg: Dataclass=None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        self.datasets[split] = FileECGDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.enable_padding,
            pad_leads=task_cfg.enable_padding_leads,
            leads_to_load=task_cfg.leads_to_load,
            label=True,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets,
            compute_mask_indices=self.cfg.precompute_mask_indices,
            leads_bucket=self.cfg.leads_bucket,
            bucket_selection=self.cfg.bucket_selection,
            training=True if 'train' in split else False,
            **self._get_mask_precompute_kwargs(task_cfg)
        )