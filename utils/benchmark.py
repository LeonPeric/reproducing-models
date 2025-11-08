import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from dataset import train_ds
from model import AlexNet


def main():
    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    device = "cuda"

    net = AlexNet().to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            for i, (X, y) in enumerate(train_loader):
                # Single copy that sets dtype + memory_format and overlaps copy
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                _ = net(X)

                if i + 1 >= 250:
                    break

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    main()
