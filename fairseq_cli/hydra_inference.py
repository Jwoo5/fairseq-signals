import logging
import os

from fairseq_signals.dataclass.initialize import add_defaults, hydra_init
from fairseq_signals.dataclass.utils import omegaconf_no_object_check
from fairseq_cli.inference import main as pre_main
from fairseq_signals import distributed_utils
from fairseq_signals.dataclass.configs import Config
from fairseq_signals.utils.utils import reset_logging

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, open_dict

logger = logging.getLogger("fairseq_cli.hydra_inference")

@hydra.main(config_path=os.path.join("..", "fairseq_signals", "config"), config_name="config")
def hydra_main(cfg: Config, **kwargs) -> None:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging() # Hydra hijacks logging, fix that
    else:
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    distributed_utils.call_main(cfg, pre_main, **kwargs)

def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"
    hydra_init(cfg_name)
    hydra_main()

if __name__ == "__main__":
    cli_main()