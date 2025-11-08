import datetime
import gc
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from torch.optim import SGD
from tqdm import tqdm

from utils import load_checkpoint, save_checkpoint


def train_model(model, train_data_loader, validation_data_loader, config, run):
    learning_rate = config.initial_learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay
    num_epochs = config.num_epochs
    patience_limit = config.patience
    device = config.device

    model.to(device)

    loss_module = nn.CrossEntropyLoss()
    optimizer = SGD(
        params=model.parameters(),
        momentum=momentum,
        weight_decay=weight_decay,
        lr=learning_rate,
    )

    start_epoch = 0
    if getattr(config, "checkpoint_path", None):
        ckpt_path = config.checkpoint_path
        ckpt = load_checkpoint(
            ckpt_path, model=model, optimizer=optimizer, map_location=device
        )
        start_epoch = ckpt.get("epoch", -1)
        best_loss_value = ckpt.get("best_loss_value", float("inf"))
        best_state = deepcopy(ckpt.get("best_state", model.state_dict()))
        learning_rate_decreases = ckpt.get("learning_rate_decreases", 0)
        patience = ckpt.get("patience", 0)

    else:
        best_loss_value = float("inf")
        best_state = deepcopy(model.state_dict())
        learning_rate_decreases = 0
        patience = 0

    run.watch(model, criterion=loss_module, log="gradients", log_freq=1, log_graph=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for data_inputs, data_labels in train_data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            logits = model(data_inputs)

            loss = loss_module(logits, data_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data_labels.size(0)
            running_loss += loss.item() * batch_size
            with torch.no_grad():
                preds = logits.argmax(1)
                running_correct += (preds == data_labels).sum().item()
                running_total += batch_size

        train_loss = running_loss / running_total
        train_top1_acc = 100.0 * running_correct / running_total

        val_loss, val_top1_acc, val_top5_acc = evaluate_tencrop(
            model, validation_data_loader, device
        )

        if val_loss <= best_loss_value:
            best_loss_value = val_loss
            best_state = deepcopy(model.state_dict())
            patience = 0
            torch.save(best_state, "model_best.pt")
            run.log_artifact("model_best.pt", type="model", name="best_model")
        else:
            patience += 1

        if patience >= patience_limit:
            learning_rate_decreases += 1
            for g in optimizer.param_groups:
                g["lr"] /= 10.0
            patience = 0

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy_top_1": train_top1_acc,
                "val/loss": val_loss,
                "val/accuracy_top_1": val_top1_acc,
                "val/accuracy_top_5": val_top5_acc,
                "best_val_loss": best_loss_value,
                "patience": patience,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rate_decreases": learning_rate_decreases,
            }
        )

        save_checkpoint(
            path="checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss_value=best_loss_value,
            best_state=best_state,
            patience=patience,
            learning_rate_decreases=learning_rate_decreases,
        )
        run.log_artifact("checkpoint.pt", type="model", name="checkpoint")

        if learning_rate_decreases > 3:
            break

    torch.save(model.state_dict(), "model_last.pt")
    run.log_artifact("model_last.pt", type="model", name="last_model")


@torch.no_grad()
def evaluate_tencrop(model, dataloader, device):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B, N, C, H, W = x.shape

        logits = model(x.view(B * N, C, H, W))
        logits = logits.view(B, N, -1)

        probs = logits.softmax(dim=-1)
        probs_mean = probs.mean(dim=1)

        log_probs_mean = (probs_mean + 1e-12).log()
        loss_sum += F.nll_loss(log_probs_mean, y, reduction="sum").item()

        topk = probs_mean.topk(5, dim=1).indices
        correct1 += (topk[:, 0] == y).sum().item()
        correct5 += (topk == y.unsqueeze(1)).any(dim=1).sum().item()
        total += B

    avg_loss = loss_sum / total
    top1_acc = 100.0 * correct1 / total
    top5_acc = 100.0 * correct5 / total
    return avg_loss, top1_acc, top5_acc
