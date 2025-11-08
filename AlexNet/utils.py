import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_loss_value,
    best_state,
    patience,
    learning_rate_decreases,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss_value": best_loss_value,
        "best_state": best_state,
        "patience": patience,
        "learning_rate_decreases": learning_rate_decreases,
    }

    torch.save(ckpt, path)


def load_checkpoint(path, *, model, optimizer=None, device=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
