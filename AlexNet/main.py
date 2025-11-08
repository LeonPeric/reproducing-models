import multiprocessing as mp
from dataclasses import asdict

import wandb
from torch.utils.data import DataLoader

from config import DatasetConfig, TrainingConfig
from dataset import get_datasets
from model import AlexNet
from train import train_model


def main():
    config = dict()
    dataset_config = DatasetConfig()
    training_config = TrainingConfig()
    config["dataset"] = asdict(dataset_config)
    config["training"] = asdict(training_config)

    run = wandb.init(project="AlexNet", config=config, name="AlexNet-FinalRun")

    train_ds, val_ds = get_datasets()

    batch_size = DatasetConfig.batch_size
    num_workers = DatasetConfig.num_workers
    pin_memory = DatasetConfig.pin_memory
    prefetch_factor = DatasetConfig.prefetch_factor
    persistent_workers = DatasetConfig.persistent_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size // 4,  # validation takes 10 crop so reduce batch size
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
    )

    net = AlexNet()
    train_model(net, train_loader, val_loader, TrainingConfig, run)

    run.finish()


if __name__ == "__main__":
    # Required for multiprocessing with DataLoader
    mp.freeze_support()
    main()
