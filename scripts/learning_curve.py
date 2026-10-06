"""
learning_curve.py
=================
Measures how segmentation accuracy depends on the NUMBER OF TRAINING PATIENTS.

Run from the project root (this takes several hours - start it before bed):

    set KMP_DUPLICATE_LIB_OK=TRUE
    venv\\Scripts\\python.exe scripts/learning_curve.py

WHY THIS EXPERIMENT
-------------------
The leave-one-hospital-out study found that the apparent cross-site accuracy
drop was NOT caused by the hospital (site effect -0.0020) but tracked the
amount of training data removed (volume effect -0.0751).

That was a negative result about sites. This experiment turns it into a
POSITIVE, quantitative result about data: it measures the actual curve of
Dice against training-set size, so the earlier drop can be predicted rather
than merely explained away.

DESIGN
------
* ONE fixed test set is used for every run - the 22 test patients from
  splits/split_patient_random.csv. Every number produced here is therefore
  directly comparable with the 0.8176 baseline.
* The remaining 88 patients form the training pool.
* Five training-set sizes are sampled from that pool: 20/40/60/80/100%.
* Each size is repeated with 3 different random seeds, so the spread caused
  by WHICH patients were drawn can be separated from the effect of HOW MANY.
* 5 sizes x 3 seeds = 15 models.

Everything is patient-level: no patient ever appears in both a training set
and the test set, and the script asserts this before each run.

RESUMABLE
---------
Results are appended to results/learning_curve.csv after every run. If the
script is interrupted, just run it again - finished runs are skipped.

OUTPUTS
-------
  results/learning_curve.csv        one row per run
  results/lc/                       model checkpoints (can be deleted later)
  splits/lc/                        the generated split files
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# This file lives in scripts/, one level below the project root. Put the
# root on sys.path (for `src`) and this folder (for sibling scripts such
# as `train`), so imports work exactly as they did when it sat at root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import BrainMRIDataset
from src.model import load_trained_model
from src.utils import get_device
from train import load_config, train_model

SOURCE_CSV = os.path.join("data", "split.csv")
REFERENCE_SPLIT = os.path.join("splits", "split_patient_random.csv")
SPLIT_DIR = os.path.join("splits", "lc")
MODEL_DIR = os.path.join("results", "lc")
RESULT_CSV = os.path.join("results", "learning_curve.csv")

FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS = [0, 1, 2]
VAL_FRACTION = 0.15
THRESHOLD = 0.5
EPS = 1e-7


def slice_dice(pred: np.ndarray, target: np.ndarray) -> float:
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + EPS) / (denom + EPS)


@torch.no_grad()
def evaluate_on_test(model, split_csv, image_size, device, batch_size):
    """Return (dice_all, dice_tumour, n_slices, n_tumour_slices)."""
    ds = BrainMRIDataset(split_csv, "test", image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    dices, has_tumour = [], []
    for images, masks in tqdm(loader, leave=False, desc="  test"):
        images = images.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs > THRESHOLD).float().cpu().numpy()
        gts = masks.numpy()
        for p, g in zip(preds, gts):
            dices.append(slice_dice(p[0], g[0]))
            has_tumour.append(bool(g[0].sum() > 0))

    d = np.array(dices)
    t = np.array(has_tumour)
    return float(d.mean()), float(d[t].mean()), len(d), int(t.sum())


def build_split(df, train_patients, val_patients, test_patients):
    out = df.copy()

    def label(p):
        if p in test_patients:
            return "test"
        if p in val_patients:
            return "val"
        if p in train_patients:
            return "train"
        return "unused"

    out["split"] = out["patient"].map(label)
    return out[out["split"] != "unused"].reset_index(drop=True)


def load_done():
    if not os.path.exists(RESULT_CSV):
        return set(), []
    prev = pd.read_csv(RESULT_CSV)
    done = set(zip(prev["n_train_patients"], prev["seed"]))
    return done, prev.to_dict("records")


def main():
    for p in (SOURCE_CSV, REFERENCE_SPLIT):
        if not os.path.exists(p):
            raise SystemExit(f"Cannot find {p}. Run from the project root.")

    cfg = load_config()
    device = get_device()
    image_size = cfg["data"]["image_size"]
    batch_size = 4

    df = pd.read_csv(SOURCE_CSV)

    # Fixed test set, taken from the baseline split so numbers stay comparable.
    ref = pd.read_csv(REFERENCE_SPLIT)
    test_patients = set(ref.loc[ref["split"] == "test", "patient"].unique())
    pool = sorted(set(df["patient"].unique()) - test_patients)

    print(f"Fixed test set : {len(test_patients)} patients")
    print(f"Training pool  : {len(pool)} patients")
    print(f"Runs to do     : {len(FRACTIONS)} sizes x {len(SEEDS)} seeds = "
          f"{len(FRACTIONS) * len(SEEDS)}\n")

    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    done, rows = load_done()
    if done:
        print(f"Resuming - {len(done)} run(s) already finished.\n")

    for frac in FRACTIONS:
        n_total = max(2, int(round(frac * len(pool))))
        for seed in SEEDS:
            rng = random.Random(1000 * seed + n_total)
            shuffled = pool[:]
            rng.shuffle(shuffled)
            chosen = shuffled[:n_total]

            n_val = max(1, int(round(VAL_FRACTION * len(chosen))))
            val_p = set(chosen[:n_val])
            train_p = set(chosen[n_val:])
            n_train = len(train_p)

            if (n_train, seed) in done:
                print(f"[skip] n_train={n_train} seed={seed} (already done)")
                continue

            tag = f"n{n_train:03d}_s{seed}"
            split_csv = os.path.join(SPLIT_DIR, f"split_{tag}.csv")
            model_path = os.path.join(MODEL_DIR, f"model_{tag}.pth")

            out = build_split(df, train_p, val_p, test_patients)
            out.to_csv(split_csv, index=False)

            # Safety: no patient in two splits, and test never leaks.
            assert out.groupby("patient")["split"].nunique().max() == 1, \
                "LEAK: a patient appears in more than one split!"
            assert not (train_p & test_patients), "LEAK: train/test overlap!"
            assert not (val_p & test_patients), "LEAK: val/test overlap!"

            n_slices = (out["split"] == "train").sum()
            print(f"\n=== n_train={n_train} patients ({n_slices} slices), "
                  f"seed={seed} ===")

            best_val, _ = train_model(
                split_csv=split_csv,
                encoder_weights=cfg["model"]["encoder_weights"],
                epochs=cfg["train"]["epochs"],
                batch_size=batch_size,
                lr=cfg["train"]["learning_rate"],
                image_size=image_size,
                num_workers=cfg["train"]["num_workers"],
                scheduler_patience=cfg["train"]["scheduler_patience"],
                scheduler_factor=cfg["train"]["scheduler_factor"],
                early_stopping_patience=cfg["train"]["early_stopping_patience"],
                best_model_path=model_path,
                curves_path=None,
                verbose=True,
            )

            model = load_trained_model(model_path, device=device)
            d_all, d_tum, n_sl, n_tum = evaluate_on_test(
                model, split_csv, image_size, device, batch_size)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            rows.append({
                "n_train_patients": n_train,
                "n_val_patients": len(val_p),
                "n_train_slices": int(n_slices),
                "seed": seed,
                "fraction": round(frac, 2),
                "best_val_dice": best_val,
                "test_dice_all": d_all,
                "test_dice_tumour": d_tum,
                "test_slices": n_sl,
                "test_tumour_slices": n_tum,
            })
            pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)
            print(f"  -> test Dice (tumour only) {d_tum:.4f}   [saved]")

    # ---------------- Summary ----------------
    res = pd.DataFrame(rows).sort_values(["n_train_patients", "seed"])
    agg = (res.groupby("n_train_patients")["test_dice_tumour"]
           .agg(["mean", "std", "count"]).reset_index())

    line = "=" * 64
    print("\n" + line)
    print("LEARNING CURVE - test Dice (tumour-bearing slices)")
    print(line)
    print(f"{'Train patients':>15}{'Mean Dice':>12}{'Std':>10}{'Runs':>7}")
    print("-" * 64)
    for _, r in agg.iterrows():
        sd = "n/a" if pd.isna(r["std"]) else f"{r['std']:.4f}"
        print(f"{int(r['n_train_patients']):>15}{r['mean']:>12.4f}"
              f"{sd:>10}{int(r['count']):>7}")
    print(line)
    print(f"\nSaved -> {RESULT_CSV}")
    print("Next: we plot this curve and read off how much of the "
          "cross-site gap it explains.")


if __name__ == "__main__":
    main()
