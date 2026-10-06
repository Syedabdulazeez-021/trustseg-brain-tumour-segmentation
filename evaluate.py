"""
evaluate.py
===========
Honest assessment on the held-out TEST patients (brains never seen in
training). Run from the project root:

    python evaluate.py                 # evaluate best model + train scratch baseline
    python evaluate.py --skip_scratch  # only evaluate the saved best model

WHAT IT REPORTS
---------------
1. Dice, IoU, sensitivity, specificity of the trained (pretrained-encoder)
   model on the test set.
2. A from-scratch U-Net (no ImageNet weights) trained for comparison, then
   the same metrics. The DIFFERENCE between the two is the measurable benefit
   of transfer learning — this is the project's headline contribution.
3. Five sample prediction overlays saved to results/ for the report/README.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# `train` now lives in scripts/, so that folder needs to be importable
# alongside the project root.
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import BrainMRIDataset
from src.metrics import all_metrics
from src.model import build_model, load_trained_model
from src.utils import denormalize, get_device, save_prediction_panel
from train import load_config, train_model


@torch.no_grad()
def evaluate_model(model, loader, device, threshold: float = 0.5) -> dict:
    """Average the four metrics over an entire loader."""
    model.eval()
    sums = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    n = 0
    for images, masks in tqdm(loader, leave=False, desc="evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        m = all_metrics(logits, masks, threshold)
        for k in sums:
            sums[k] += m[k]
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def save_samples(model, dataset, device, out_dir: str, n: int = 5,
                 threshold: float = 0.5):
    """Save up to `n` sample 3-panel overlays, preferring slices with tumour."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # Prefer slices that actually contain a tumour so the overlay is meaningful.
    chosen, idx = [], 0
    while len(chosen) < n and idx < len(dataset):
        _, mask = dataset[idx]
        if mask.sum() > 0:
            chosen.append(idx)
        idx += 1
    # Top up with any remaining indices if not enough tumour slices.
    for i in range(len(dataset)):
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)

    for j, i in enumerate(chosen[:n]):
        image_t, mask_t = dataset[i]
        logits = model(image_t.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob > threshold).astype("float32")

        rgb = denormalize(image_t)
        true_mask = mask_t[0].numpy()
        out_path = os.path.join(out_dir, f"sample_prediction_{j + 1}.png")
        save_prediction_panel(rgb, true_mask, pred, out_path)
        print(f"  saved {out_path}")


def print_comparison_table(pretrained: dict, scratch: dict | None):
    """Pretty-print the metrics comparison table."""
    metrics = ["dice", "iou", "sensitivity", "specificity"]
    header = f"{'Metric':<14}{'Pretrained(resnet34)':>22}"
    if scratch is not None:
        header += f"{'From-scratch':>16}{'Improvement':>14}"
    print("\n" + "=" * len(header))
    print("TEST-SET RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<14}{pretrained[m]:>22.4f}"
        if scratch is not None:
            diff = pretrained[m] - scratch[m]
            row += f"{scratch[m]:>16.4f}{diff:>+14.4f}"
        print(row)
    print("=" * len(header))


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate on the test set.")
    parser.add_argument("--split_csv", default=cfg["data"]["split_csv"])
    parser.add_argument("--model_path", default=cfg["train"]["best_model_path"])
    parser.add_argument("--skip_scratch", action="store_true",
                        help="Skip training the from-scratch baseline.")
    parser.add_argument("--scratch_epochs", type=int,
                        default=cfg["train"]["epochs"],
                        help="Epochs to train the from-scratch baseline.")
    args = parser.parse_args()

    device = get_device()
    image_size = cfg["data"]["image_size"]
    results_dir = cfg["train"]["results_dir"]

    test_ds = BrainMRIDataset(args.split_csv, "test", image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False)
    print(f"Test slices: {len(test_ds)} | Device: {device}")

    # --- 1. Evaluate the trained (pretrained-encoder) model ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No trained model at {args.model_path}. Run train.py first."
        )
    print(f"\nLoading trained model: {args.model_path}")
    model = load_trained_model(args.model_path, device=device)
    pretrained_metrics = evaluate_model(model, test_loader, device)
    print("Pretrained-encoder metrics:",
          {k: round(v, 4) for k, v in pretrained_metrics.items()})

    # --- 2. Save sample overlays ---
    print("\nSaving sample prediction overlays...")
    save_samples(model, test_ds, device, results_dir, n=5)

    # --- 3. From-scratch baseline for comparison ---
    scratch_metrics = None
    if not args.skip_scratch:
        print("\nTraining a from-scratch U-Net (no ImageNet weights) for "
              "comparison...")
        scratch_path = os.path.join(results_dir, "scratch_model.pth")
        train_model(
            split_csv=args.split_csv,
            encoder_weights=None,
            epochs=args.scratch_epochs,
            batch_size=cfg["train"]["batch_size"],
            lr=cfg["train"]["learning_rate"],
            image_size=image_size,
            num_workers=cfg["train"]["num_workers"],
            scheduler_patience=cfg["train"]["scheduler_patience"],
            scheduler_factor=cfg["train"]["scheduler_factor"],
            early_stopping_patience=cfg["train"]["early_stopping_patience"],
            best_model_path=scratch_path,
            curves_path=os.path.join(results_dir, "scratch_curves.png"),
            verbose=True,
        )
        scratch_model = load_trained_model(scratch_path, device=device)
        scratch_metrics = evaluate_model(scratch_model, test_loader, device)
        print("From-scratch metrics:",
              {k: round(v, 4) for k, v in scratch_metrics.items()})

    # --- 4. Comparison table ---
    print_comparison_table(pretrained_metrics, scratch_metrics)


if __name__ == "__main__":
    main()
