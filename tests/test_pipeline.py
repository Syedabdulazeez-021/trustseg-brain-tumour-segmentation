"""
test_pipeline.py
================
A FAST, synthetic end-to-end smoke test. It needs NO download and finishes
in well under a minute on CPU. Run it BEFORE committing to a 2-hour training
run, to be sure every moving part connects.

    python tests/test_pipeline.py        # plain run
    pytest tests/test_pipeline.py        # also works under pytest

What it checks
--------------
1. model.py builds and does a forward pass (random encoder, no network).
2. losses.py gives lower loss for correct predictions.
3. metrics.py returns sensible values.
4. dataset.py + prepare_data.py build a CSV and load tensors (incl. empty mask).
5. train.py.train_model() actually trains for 2 epochs on tiny fake data and
   reports a validation Dice.
6. evaluate.py.evaluate_model() runs on the fake test split.

The "data" is a handful of generated images where a bright square in the
image lines up with a white square in the mask — easy enough that even a
2-epoch toy run shows the loss going down.
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch

# Make project root importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import BrainMRIDataset                       # noqa: E402
from src.losses import DiceBCELoss                            # noqa: E402
from src.metrics import all_metrics                           # noqa: E402
from src.model import build_model, count_parameters           # noqa: E402
from data.prepare_data import build_split_csv                 # noqa: E402


def _make_fake_dataset(root: str, n_patients: int = 6, slices: int = 4):
    """Create a tiny TCGA-style dataset of synthetic slices + masks."""
    raw = os.path.join(root, "kaggle_3m")
    os.makedirs(raw, exist_ok=True)
    rng = np.random.default_rng(0)
    for p in range(n_patients):
        pdir = os.path.join(raw, f"TCGA_FAKE_{p:02d}")
        os.makedirs(pdir, exist_ok=True)
        for s in range(1, slices + 1):
            img = (rng.random((64, 64, 3)) * 100).astype(np.uint8)
            mask = np.zeros((64, 64), np.uint8)
            # Half the slices have a tumour; the other half are empty.
            if s % 2 == 0:
                y, x = rng.integers(10, 30, size=2)
                mask[y:y + 20, x:x + 20] = 255
                img[y:y + 20, x:x + 20] = 230  # bright square = learnable cue
            cv2.imwrite(os.path.join(pdir, f"TCGA_FAKE_{p:02d}_{s}.tif"), img)
            cv2.imwrite(os.path.join(pdir, f"TCGA_FAKE_{p:02d}_{s}_mask.tif"), mask)
    return raw


def test_model_builds():
    net = build_model(encoder_weights=None)
    out = net(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1, 256, 256)
    assert count_parameters(net) > 1_000_000
    print("[1/6] model builds + forward pass OK")


def test_loss_behaviour():
    crit = DiceBCELoss()
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1.0
    good = crit(torch.where(target > 0.5, 10.0, -10.0), target).item()
    bad = crit(torch.where(target > 0.5, -10.0, 10.0), target).item()
    assert good < bad
    print(f"[2/6] loss OK (correct={good:.3f} < wrong={bad:.3f})")


def test_metrics_behaviour():
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 16:48, 16:48] = 1.0
    perfect = torch.where(target > 0.5, 10.0, -10.0)
    m = all_metrics(perfect, target)
    assert m["dice"] > 0.99 and m["iou"] > 0.99
    print(f"[3/6] metrics OK (dice={m['dice']:.3f}, iou={m['iou']:.3f})")


def test_dataset_and_split(tmp_root):
    raw = _make_fake_dataset(tmp_root)
    csv_path = os.path.join(tmp_root, "split.csv")
    df = build_split_csv(raw, csv_path, n_val=2, n_test=2, seed=0)

    # No patient may appear in two splits.
    by_split = df.groupby("patient")["split"].nunique()
    assert (by_split == 1).all(), "A patient leaked across splits!"

    ds = BrainMRIDataset(csv_path, "train", image_size=64)
    img_t, msk_t = ds[0]
    assert img_t.shape == (3, 64, 64) and msk_t.shape == (1, 64, 64)
    assert set(torch.unique(msk_t).tolist()).issubset({0.0, 1.0})
    print(f"[4/6] dataset + patient-level split OK "
          f"({len(df)} slices, no leakage)")
    return csv_path


def test_train_and_eval(csv_path):
    from train import train_model
    from evaluate import evaluate_model
    from torch.utils.data import DataLoader
    from src.utils import get_device

    best_path = os.path.join(os.path.dirname(csv_path), "toy_model.pth")
    best_dice, history = train_model(
        split_csv=csv_path,
        encoder_weights=None,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        image_size=64,
        num_workers=0,
        scheduler_patience=2,
        scheduler_factor=0.5,
        early_stopping_patience=5,
        best_model_path=best_path,
        curves_path=None,
        verbose=False,
    )
    assert len(history["val_dice"]) == 2
    assert os.path.exists(best_path)
    print(f"[5/6] train_model ran 2 epochs OK (best val Dice={best_dice:.3f})")

    device = get_device()
    from src.model import load_trained_model
    model = load_trained_model(best_path, device=device)
    test_ds = BrainMRIDataset(csv_path, "test", image_size=64)
    loader = DataLoader(test_ds, batch_size=4)
    metrics = evaluate_model(model, loader, device)
    assert set(metrics) == {"dice", "iou", "sensitivity", "specificity"}
    print(f"[6/6] evaluate_model ran OK on test split "
          f"(dice={metrics['dice']:.3f})")


def run_all():
    print("=" * 60)
    print("END-TO-END SYNTHETIC SMOKE TEST (no dataset needed)")
    print("=" * 60)
    test_model_builds()
    test_loss_behaviour()
    test_metrics_behaviour()
    tmp_root = tempfile.mkdtemp()
    try:
        csv_path = test_dataset_and_split(tmp_root)
        test_train_and_eval(csv_path)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("\nALL CHECKS PASSED — the full pipeline is wired correctly.")
    print("You can now safely run real training on the LGG dataset.")


# --- pytest fixtures (optional; only used when run under pytest) ---
def test_smoke():
    """Single entry point so `pytest` exercises the whole pipeline."""
    run_all()


if __name__ == "__main__":
    run_all()
