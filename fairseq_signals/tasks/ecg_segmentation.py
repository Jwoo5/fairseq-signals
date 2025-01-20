import logging
import os

from dataclasses import dataclass, field

from fairseq_signals.data import SegmentationECGDataset
from fairseq_signals.dataclass import Dataclass
from fairseq_signals.tasks.ecg_pretraining import ECGPretrainingTask, ECGPretrainingConfig

from . import register_task


logger = logging.getLogger(__name__)

@dataclass
class ECGSegmentationConfig(ECGPretrainingConfig):
    apply_cgm: bool = field(
        default=False,
        metadata={
            "help": "whether to apply classification-guided module (CGM) along with segmentation. "
                "if set, each sample should contain `label` as a key for classification label"
        }
    )

@register_task("ecg_segmentation", dataclass=ECGSegmentationConfig)
class ECGSegmentationTask(ECGPretrainingTask):
    cfg: ECGSegmentationConfig

    def __init__(
        self,
        cfg: ECGSegmentationConfig
    ):
        super().__init__(cfg)

        self.apply_cgm = cfg.apply_cgm
    
    @classmethod
    def setup_task(cls, cfg: ECGSegmentationConfig, **kwargs):
        """
        Setup the task
        
        Args:
            cfg (ECGSegmentationConfig): configuration of this task
        """
        return cls(cfg)

    def _get_perturbation_kwargs(self):
        return {
            "p": self.cfg.p,
            "max_amplitude": self.cfg.max_amplitude,
            "min_amplitude": self.cfg.min_amplitude,
            "dependency": self.cfg.dependency,
            "shift_ratio": self.cfg.shift_ratio,
            "num_segment": self.cfg.num_segment,
            "max_freq": self.cfg.max_freq,
            "min_freq": self.cfg.min_freq,
            "freq": self.cfg.sample_rate,
            "k": self.cfg.k,
            "mask_leads_selection": self.cfg.mask_leads_selection,
            "mask_leads_prob": self.cfg.mask_leads_prob,
            "mask_leads_condition": self.cfg.mask_leads_condition,
        }

    def load_dataset(self, split: str, task_cfg: Dataclass=None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        manifest_path = os.path.join(data_path, f"{split}.tsv")

        self.datasets[split] = SegmentationECGDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            perturbation_mode=self.cfg.perturbation_mode,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.enable_padding,
            label=self.apply_cgm,
            normalize=task_cfg.normalize,
            mean_path=task_cfg.get("mean_path", self.cfg.mean_path),
            std_path=task_cfg.get("std_path", self.cfg.std_path),
            num_buckets=self.cfg.num_batch_buckets,
            training=True if "train" in split else False,
            **self._get_perturbation_kwargs()
        )