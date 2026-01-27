"""
Script to download all datasets defined in the benchmark configuration.
This is a minimal script that only iterates over datasets to ensure they are downloaded.
"""

import logging

import hydra
import omegaconf
from hydra.utils import instantiate
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

# Force the execution of __init__.py if this file is executed directly.
import model_merging  # noqa
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.io_utils import load_model_from_hf

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """
    Iterate over all datasets in the benchmark to ensure they are downloaded.

    Args:
        cfg: run configuration, defined by Hydra in /conf
    """
    seed_index_everything(cfg)

    num_tasks = len(cfg.benchmark.datasets)
    pylogger.info(f"Number of datasets to download: {num_tasks}")

    # Load encoder to get the preprocessing function
    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    for i, dataset_cfg in enumerate(cfg.benchmark.datasets, 1):
        pylogger.info(f"[{i}/{num_tasks}] Downloading/verifying dataset: {dataset_cfg.name}")
        
        # Instantiating the dataset will trigger the download if not already present
        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess
        )
        
        pylogger.info(f"[{i}/{num_tasks}] Dataset {dataset_cfg.name} ready. Classes: {len(dataset.classnames)}")

    pylogger.info("All datasets downloaded successfully!")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="multitask.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
