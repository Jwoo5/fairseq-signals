import logging
import os

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional

from fairseq_signals.data import FileECGDataset, PathECGDataset
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks.ecg_pretraining import ECGPretrainingTask, ECGPretrainingConfig

from . import register_task

logger = logging.getLogger(__name__)

@dataclass
class ECGClassificationConfig(ECGPretrainingConfig):
    path_dataset: bool = field(
        default=False,
        metadata={
            "help": "whether to load dataset based on file path, instead of direct file in itself. "
            "note that this dataset is also used for multi head classification"
        }
    )
    label_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "a separate .npy file path for loading labels for the input data. "
                "if set, it assumes that each input data has an additional field 'idx' which "
                "indicates the corresponding index within the numpy array for the labels."
        }
    )
    load_specific_lead: bool = field(
        default=False,
        metadata={
            "help": "if set true, load specific lead for each ecg "
            "note that `lead` key should exist in the data file"
        }
    )

@register_task("ecg_classification", dataclass=ECGClassificationConfig)
class ECGDiagnosisTask(ECGPretrainingTask):
    cfg: ECGClassificationConfig

    def __init__(
        self,
        cfg: ECGClassificationConfig
    ):
        super().__init__(cfg)

        self.path_dataset = cfg.path_dataset
        self.label_file = cfg.label_file
        self.load_specific_lead = cfg.load_specific_lead
    
    @classmethod
    def setup_task(cls, cfg: ECGClassificationConfig, **kwargs):
        """Setup the task
        
        Args:
            cfg (ECGDiagnosisConfig): configuration of this task
        """
        return cls(cfg)
    
    def load_dataset(self, split: str, task_cfg: Dataclass=None, label=True, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        if not self.path_dataset:
            self.datasets[split] = FileECGDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                pad_leads=task_cfg.enable_padding_leads,
                leads_to_load=task_cfg.leads_to_load,
                label=label,
                label_file=self.cfg.label_file,
                filter=task_cfg.filter,
                normalize=task_cfg.normalize,
                mean_path=task_cfg.get("mean_path", self.cfg.mean_path),
                std_path=task_cfg.get("std_path", self.cfg.std_path),
                num_buckets=self.cfg.num_batch_buckets,
                compute_mask_indices=self.cfg.precompute_mask_indices,
                leads_bucket=self.cfg.leads_bucket,
                bucket_selection=self.cfg.bucket_selection,
                training=True if 'train' in split else False,
                **self._get_mask_precompute_kwargs(task_cfg),
                **kwargs,
            )
        else:
            self.datasets[split] = PathECGDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                load_specific_lead=self.load_specific_lead,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.enable_padding,
                pad_leads=task_cfg.enable_padding_leads,
                leads_to_load=task_cfg.leads_to_load,
                label=label,
                label_file=self.cfg.label_file,
                filter=task_cfg.filter,
                normalize=task_cfg.normalize,
                mean_path=task_cfg.get("mean_path", self.cfg.mean_path),
                std_path=task_cfg.get("std_path", self.cfg.std_path),
                num_buckets=self.cfg.num_batch_buckets,
                compute_mask_indices=self.cfg.precompute_mask_indices,
                leads_bucket=self.cfg.leads_bucket,
                bucket_selection=self.cfg.bucket_selection,
                training=True if 'train' in split else False,
                **self._get_mask_precompute_kwargs(task_cfg),
                **kwargs,
            )