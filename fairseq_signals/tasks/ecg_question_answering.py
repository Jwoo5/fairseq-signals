from dataclasses import dataclass
import logging
import os

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING

from fairseq_signals.data import FileECGQADataset
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks.ecg_text_pretaining import ECGTextPretrainingConfig, ECGTextPretrainingTask

from . import register_task

logger = logging.getLogger(__name__)

@dataclass
class ECGQuestionAnsweringConfig(ECGTextPretrainingConfig):
    json_dataset: bool = field(
        default=True,
        metadata={
            'help': 'if true, load json dataset (default)'
        }
    )

@register_task('ecg_question_answering', dataclass=ECGQuestionAnsweringConfig)
class ECGQuestionAnsweringTask(ECGTextPretrainingTask):
    cfg: ECGQuestionAnsweringConfig
    
    def __init__(self, cfg: ECGQuestionAnsweringConfig):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg: ECGQuestionAnsweringConfig, **kwargs):
        """Setup the task
        
        Args:
            cfg (ECGDiagnosisConfig): configuration of this task
        """
        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: Dataclass=None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = FileECGQADataset(
            manifest_path,
            pad_token_id=task_cfg.pad_token,
            sep_token_id=task_cfg.sep_token,
            pad=task_cfg.enable_padding,
            sample_rate=task_cfg.get('sample_rate', self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            max_text_size=self.cfg.max_text_size,
            filter=task_cfg.filter,
            normalize=task_cfg.normalize,
            mean_path=task_cfg.get("mean_path", self.cfg.mean_path),
            std_path=task_cfg.get("std_path", self.cfg.std_path),
            training=True if 'train' in split else False,
        )