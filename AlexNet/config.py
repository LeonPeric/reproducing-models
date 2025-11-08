from dataclasses import dataclass


@dataclass
class DatasetConfig:
    root_data_path: str = "./data/shards"
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    test_data_path: str = "./data/test"
    val_ground_truth_path: str = "./data/val_ground_truth.txt"
    test_ground_truth_path: str = "./data/test_ground_truth.txt"
    mapping_path: str = "./data/mapping.json"

    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class TrainingConfig:
    initial_learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_epochs: int = 200
    patience: int = 30
    device = "cuda"
    checkpoint_path: str | None = "models/checkpoint.pt"
