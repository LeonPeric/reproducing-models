import json
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from config import DatasetConfig

MEAN = (0.485, 0.456, 0.406)
STD = (1, 1, 1)
INITIAL_IMAGE_SIZE = 256
CROP_SIZE = 227


class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, dtype=torch.float32)
        self.eigvec = torch.tensor(eigvec, dtype=torch.float32)

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        alpha = torch.randn(3) * self.alphastd
        rgb = (self.eigvec @ (self.eigval * alpha)).view(3, 1, 1)
        return img + rgb


lighting = Lighting(
    alphastd=0.1,
    eigval=[0.2175, 0.0188, 0.0045],
    eigvec=[
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ],
)


def apply_lighting(x):
    return lighting(x)


def apply_ten_crop_transform(crops):
    tensors = []
    for c in crops:
        if isinstance(c, torch.Tensor):
            tensors.append(c)
        else:
            tensors.append(T.ToTensor()(c))

    return torch.stack(tensors)


train_transform = T.Compose(
    [
        T.Resize(INITIAL_IMAGE_SIZE),
        T.CenterCrop(INITIAL_IMAGE_SIZE),
        T.RandomCrop(CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Lambda(apply_lighting),
        T.Normalize(MEAN, STD),
    ]
)

test_transform = T.Compose(
    [
        T.Resize(INITIAL_IMAGE_SIZE),
        T.CenterCrop(INITIAL_IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.TenCrop(CROP_SIZE),
        T.Lambda(apply_ten_crop_transform),
    ]
)


class ListDataset(Dataset):
    def __init__(self, root, labels_txt, transform=None, label_base=0):
        self.root = Path(root)
        self.transform = transform
        self.label_base = int(label_base)

        with open(labels_txt, "r") as f:
            raw = [ln.strip() for ln in f if ln.strip()]
        self.labels = [int(x) - self.label_base for x in raw]

        imgs = [p for p in self.root.iterdir() if p.suffix == ".JPEG"]

        def numeric_suffix_key(p: Path):
            m = re.search(r"(\d+)(?=\.[^.]+$)", p.name)
            if m:
                return (int(m.group(1)), p.name.lower())
            return (float("inf"), p.name.lower())

        imgs.sort(key=numeric_suffix_key)
        self.images = imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        target = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_datasets():
    mapping = json.load(open(DatasetConfig.mapping_path))

    train_ds = ImageFolder(DatasetConfig.train_data_path, transform=train_transform)
    train_ds.class_to_idx = {v: int(k) - 1 for k, v in mapping.items()}
    train_ds.samples = [
        (p, train_ds.class_to_idx[Path(p).parent.name]) for p, _ in train_ds.samples
    ]
    train_ds.targets = [t for _, t in train_ds.samples]

    val_ds = ListDataset(
        root=DatasetConfig.val_data_path,
        labels_txt=DatasetConfig.val_ground_truth_path,
        transform=test_transform,
        label_base=1,
    )

    test_ds = ListDataset(
        root=DatasetConfig.test_data_path,
        labels_txt=DatasetConfig.test_ground_truth_path,
        transform=test_transform,
        label_base=1,
    )

    return train_ds, val_ds, test_ds
