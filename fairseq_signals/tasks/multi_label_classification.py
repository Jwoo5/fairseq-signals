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
class MultiLabelClassificationConfig(ECGPretrainingConfig):
    num_labels: int = field(
        default=MISSING, metadata={"help": "number of labels to be classified"}
    )

@register_task("multi_label_classification", dataclass=MultiLabelClassificationConfig)
class MultiLabelClassificationTask(ECGPretrainingTask):
    cfg: MultiLabelClassificationConfig

    def __init__(
        self,
        cfg: MultiLabelClassificationConfig
    ):
        super().__init__(cfg)
    
    @classmethod
    def setup_task(cls, cfg: MultiLabelClassificationConfig, **kwargs):
        """Setup the task
        
        Args:
            cfg (MultiLabelClassificationConfig): configuration of this task
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