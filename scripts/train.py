"""
train.py
========
The complete training pipeline. Run from the project root:

    python scripts/train.py                       # uses config/config.yaml
    python scripts/train.py --epochs 5            # quick override
    python scripts/train.py --encoder_weights none  # from-scratch baseline

MULTI-INSTITUTION STUDY
-----------------------
Use --split_csv to point at a generated split, and --tag to keep the outputs
of each run separate:

    python scripts/train.py --split_csv splits/split_patient_random.csv --tag random
    python scripts/train.py --split_csv splits/split_holdout_DU.csv     --tag DU

With --tag DU the outputs become results/best_model_DU.pth and
results/curves_DU.png. Without --tag the original filenames are used, so your
existing model is never overwritten.

WHAT TRAINING DOES (analogy)
----------------------------
Training is like a student doing thousands of practice questions. For each
batch of MRI slices the model guesses the tumour, we measure how wrong it is
(the loss), and the optimiser nudges every weight a little to be less wrong
next time. After each full pass over the data (an "epoch") we test on the
validation patients to see if it is actually getting better at brains it has
not memorised. We keep the version with the best validation Dice.

KEY FEATURES
------------
* Auto-detects GPU vs CPU.
* Mixed precision (faster, less memory) when a GPU is present.
* ReduceLROnPlateau: lowers the learning rate when val Dice stalls.
* Early stopping: stops if val Dice has not improved for `patience` epochs.
* Saves the best model and a training-curves PNG.
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make "src" importable when running from the project root.
# This file lives in scripts/, one level below the project root. Put the
# root on sys.path (for `src`) and this folder (for sibling scripts such
# as `train`), so imports work exactly as they did when it sat at root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import BrainMRIDataset
from src.losses import DiceBCELoss
from src.metrics import dice_coefficient
from src.model import build_model
from src.utils import get_device, save_training_curves


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_one_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Run a single epoch. If `optimizer` is None -> evaluation mode.

    Returns (mean_loss, mean_dice) over the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    use_amp = scaler is not None and device.type == "cuda"

    progress = tqdm(loader, leave=False, desc="train" if is_train else "val")
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_dice = dice_coefficient(logits.detach(), masks)
        total_loss += loss.item()
        total_dice += batch_dice
        n_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


def train_model(
    split_csv: str,
    encoder_weights,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    num_workers: int,
    scheduler_patience: int,
    scheduler_factor: float,
    early_stopping_patience: int,
    best_model_path: str,
    curves_path: str | None = None,
    verbose: bool = True,
):
    """Train a U-Net and return (best_val_dice, history).

    `encoder_weights`="imagenet" for transfer learning, or None for the
    from-scratch baseline. This function is also called by evaluate.py.
    """
    device = get_device()
    if verbose:
        print(f"Device: {device}")
        print(f"Encoder weights: {encoder_weights}")

    train_ds = BrainMRIDataset(split_csv, "train", image_size)
    val_ds = BrainMRIDataset(split_csv, "val", image_size)
    if verbose:
        print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    model = build_model(encoder_weights=encoder_weights).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor,
        patience=scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val_dice = -1.0
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice = run_one_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        with torch.no_grad():
            va_loss, va_dice = run_one_epoch(
                model, val_loader, criterion, device
            )

        scheduler.step(va_dice)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)

        cur_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
                  f"val loss {va_loss:.4f} dice {va_dice:.4f} | lr {cur_lr:.1e}")

        # Save best + early stopping (monitor val Dice).
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            epochs_without_improve = 0
            os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)
            torch.save({"model_state": model.state_dict(),
                        "val_dice": best_val_dice,
                        "encoder_weights": encoder_weights},
                       best_model_path)
            if verbose:
                print(f"  -> new best val Dice {best_val_dice:.4f}; saved "
                      f"{best_model_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val-Dice improvement for "
                          f"{early_stopping_patience} epochs).")
                break

    if curves_path:
        save_training_curves(history, curves_path)
        if verbose:
            print(f"Training curves saved -> {curves_path}")

    if verbose:
        print(f"\nBEST VALIDATION DICE: {best_val_dice:.4f}")
    return best_val_dice, history


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train the U-Net.")
    parser.add_argument("--split_csv", default=cfg["data"]["split_csv"])
    parser.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
    parser.add_argument("--batch_size", type=int,
                        default=cfg["train"]["batch_size"])
    parser.add_argument("--lr", type=float,
                        default=cfg["train"]["learning_rate"])
    parser.add_argument(
        "--encoder_weights", default=cfg["model"]["encoder_weights"],
        help="'imagenet' for transfer learning, or 'none' for from scratch.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Suffix for output filenames, e.g. --tag DU saves "
             "results/best_model_DU.pth. Leave empty to use the config paths.",
    )
    args = parser.parse_args()

    enc = None if str(args.encoder_weights).lower() in ("none", "null", "") \
        else args.encoder_weights

    # Keep every run's outputs separate when --tag is given.
    base_model_path = cfg["train"]["best_model_path"]
    if args.tag:
        root, ext = os.path.splitext(base_model_path)
        best_model_path = f"{root}_{args.tag}{ext}"
        curves_name = f"curves_{args.tag}.png"
    else:
        best_model_path = base_model_path
        curves_name = "curves.png"

    print(f"Split CSV : {args.split_csv}")
    print(f"Model out : {best_model_path}")

    train_model(
        split_csv=args.split_csv,
        encoder_weights=enc,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=cfg["data"]["image_size"],
        num_workers=cfg["train"]["num_workers"],
        scheduler_patience=cfg["train"]["scheduler_patience"],
        scheduler_factor=cfg["train"]["scheduler_factor"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
        best_model_path=best_model_path,
        curves_path=os.path.join(cfg["train"]["results_dir"], curves_name),
    )


if __name__ == "__main__":
    main()