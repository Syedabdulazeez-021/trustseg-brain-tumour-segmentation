"""
evaluate_domain_shift.py  (v2 - includes matched controls)
==========================================================
Evaluates all NINE trained models on their held-out TEST patients and prints
the decisive comparison for the multi-institution study.

Run from the project root:

    venv\\Scripts\\python.exe scripts/evaluate_domain_shift.py

THE EXPERIMENTS
---------------
  baseline      random mixed-hospital split (conventional protocol)
  LOHO x4       DU / HT / CS / FG entirely absent from training
  control x4    same patient counts as each LOHO split, but the test
                patients are sampled across all hospitals

THE QUESTION THIS ANSWERS
-------------------------
A LOHO model scores lower than baseline. But it also trained on fewer
patients. Was the drop caused by the missing HOSPITAL, or merely by the
missing DATA?

  If control ~= baseline while LOHO stays low  -> domain shift is real.
  If control falls as far as LOHO              -> it was training volume.

The "site effect" column below is the part attributable to the hospital
after the data-volume effect has been subtracted:

    site_effect = dice_LOHO - dice_control      (both same training size)

OUTPUTS
-------
  results/domain_shift_summary.csv      one row per experiment
  results/domain_shift_per_patient.csv  one row per patient
  results/domain_shift_paired.csv       LOHO vs control, per site

NOTE ON STATISTICS
------------------
LOHO and control test sets contain DIFFERENT patients, so this is an
independent-samples comparison with matched sample sizes, not a paired test.
A Mann-Whitney U test is reported per site when scipy is available.
"""

import os
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
from train import load_config

SITES = ["DU", "HT", "CS", "FG"]

EXPERIMENTS = [
    ("random", "splits/split_patient_random.csv", "baseline",
     "Mixed hospitals (conventional split)"),
]
for _s in SITES:
    EXPERIMENTS.append(
        (_s, f"splits/split_holdout_{_s}.csv", "loho",
         f"Unseen hospital: {_s}"))
for _s in SITES:
    EXPERIMENTS.append(
        (f"ctrl{_s}", f"splits/split_control_{_s}.csv", "control",
         f"Matched control for {_s} (mixed test)"))

THRESHOLD = 0.5
EPS = 1e-7


def slice_dice(pred: np.ndarray, target: np.ndarray) -> float:
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + EPS) / (denom + EPS)


@torch.no_grad()
def per_slice_scores(model, split_csv, image_size, device, batch_size):
    ds = BrainMRIDataset(split_csv, "test", image_size)

    rows = pd.read_csv(split_csv)
    rows = rows[rows["split"] == "test"].reset_index(drop=True)
    if len(rows) != len(ds):
        raise SystemExit(
            f"Row mismatch for {split_csv}: CSV has {len(rows)} test rows but "
            f"the dataset built {len(ds)}. Cannot align patients."
        )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    dices, has_tumour = [], []
    for images, masks in tqdm(loader, leave=False, desc="  evaluating"):
        images = images.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs > THRESHOLD).float().cpu().numpy()
        gts = masks.numpy()
        for p, g in zip(preds, gts):
            p2, g2 = p[0], g[0]
            dices.append(slice_dice(p2, g2))
            has_tumour.append(bool(g2.sum() > 0))

    return pd.DataFrame({
        "patient": rows["patient"].values,
        "dice": dices,
        "has_tumour": has_tumour,
    })


def main():
    cfg = load_config()
    device = get_device()
    image_size = cfg["data"]["image_size"]
    results_dir = cfg["train"]["results_dir"]
    root, ext = os.path.splitext(cfg["train"]["best_model_path"])
    batch_size = 4

    print(f"Device: {device}\n")

    summary_rows, per_patient_frames, patient_lookup = [], [], {}

    for tag, split_csv, kind, desc in EXPERIMENTS:
        model_path = f"{root}_{tag}{ext}"
        if not os.path.exists(model_path):
            print(f"[skip] {tag}: no model at {model_path}")
            continue
        if not os.path.exists(split_csv):
            print(f"[skip] {tag}: no split at {split_csv}")
            continue

        print(f"{tag:<9} {desc}")
        model = load_trained_model(model_path, device=device)
        df = per_slice_scores(model, split_csv, image_size, device, batch_size)
        df["experiment"] = tag

        tumour_only = df[df["has_tumour"]]

        pp = (tumour_only.groupby("patient")["dice"].mean()
              .reset_index().rename(columns={"dice": "dice_tumour"}))
        pp["experiment"] = tag
        pp["kind"] = kind
        per_patient_frames.append(pp)
        patient_lookup[tag] = pp["dice_tumour"].values

        summary_rows.append({
            "experiment": tag,
            "kind": kind,
            "description": desc,
            "n_patients": df["patient"].nunique(),
            "n_slices": len(df),
            "n_tumour_slices": len(tumour_only),
            "dice_all": df["dice"].mean(),
            "dice_tumour": tumour_only["dice"].mean(),
            "dice_tumour_patient_mean": pp["dice_tumour"].mean(),
            "dice_tumour_patient_std": pp["dice_tumour"].std(),
        })

        print(f"          Dice (all slices)   {summary_rows[-1]['dice_all']:.4f}")
        print(f"          Dice (tumour only)  {summary_rows[-1]['dice_tumour']:.4f}\n")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not summary_rows:
        raise SystemExit("Nothing evaluated. Have all nine models been trained?")

    summary = pd.DataFrame(summary_rows)

    base = None
    if "random" in summary["experiment"].values:
        base = summary.loc[summary["experiment"] == "random",
                           "dice_tumour"].iloc[0]
        summary["gap_vs_baseline"] = summary["dice_tumour"] - base
    else:
        summary["gap_vs_baseline"] = np.nan

    os.makedirs(results_dir, exist_ok=True)
    summary.to_csv(os.path.join(results_dir, "domain_shift_summary.csv"),
                   index=False)
    pd.concat(per_patient_frames).to_csv(
        os.path.join(results_dir, "domain_shift_per_patient.csv"), index=False)

    # ---------------- Table 1: every experiment ----------------
    line = "=" * 86
    print("\n" + line)
    print("TABLE 1 - ALL EXPERIMENTS")
    print(line)
    print(f"{'Experiment':<12}{'Type':<10}{'Patients':>9}{'Slices':>8}"
          f"{'Dice(all)':>11}{'Dice(tum)':>11}{'vs base':>10}")
    print("-" * 86)
    for _, r in summary.iterrows():
        gap = "" if pd.isna(r["gap_vs_baseline"]) else f"{r['gap_vs_baseline']:+.4f}"
        print(f"{r['experiment']:<12}{r['kind']:<10}{r['n_patients']:>9}"
              f"{r['n_slices']:>8}{r['dice_all']:>11.4f}"
              f"{r['dice_tumour']:>11.4f}{gap:>10}")
    print(line)

    # ---------------- Table 2: the decisive comparison ----------------
    try:
        from scipy.stats import mannwhitneyu
        have_scipy = True
    except ImportError:
        have_scipy = False

    paired_rows = []
    for s in SITES:
        if s not in summary["experiment"].values:
            continue
        c = f"ctrl{s}"
        if c not in summary["experiment"].values:
            continue

        d_loho = summary.loc[summary["experiment"] == s, "dice_tumour"].iloc[0]
        d_ctrl = summary.loc[summary["experiment"] == c, "dice_tumour"].iloc[0]

        p = np.nan
        if have_scipy:
            try:
                p = mannwhitneyu(patient_lookup[s], patient_lookup[c],
                                 alternative="two-sided").pvalue
            except ValueError:
                p = np.nan

        paired_rows.append({
            "site": s,
            "n_test_patients": int(summary.loc[summary["experiment"] == s,
                                               "n_patients"].iloc[0]),
            "dice_loho": d_loho,
            "dice_control": d_ctrl,
            "site_effect": d_loho - d_ctrl,
            "volume_effect": (d_ctrl - base) if base is not None else np.nan,
            "p_value": p,
        })

    if paired_rows:
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(os.path.join(results_dir, "domain_shift_paired.csv"),
                      index=False)

        print("\n" + line)
        print("TABLE 2 - DECISIVE COMPARISON: is the drop the hospital, or the data?")
        print(line)
        print(f"{'Site':<7}{'N':>5}{'Dice LOHO':>12}{'Dice ctrl':>12}"
              f"{'SITE effect':>13}{'volume effect':>15}{'p':>10}")
        print("-" * 86)
        for _, r in paired.iterrows():
            pv = "n/a" if pd.isna(r["p_value"]) else f"{r['p_value']:.4f}"
            ve = "" if pd.isna(r["volume_effect"]) else f"{r['volume_effect']:+.4f}"
            print(f"{r['site']:<7}{r['n_test_patients']:>5}"
                  f"{r['dice_loho']:>12.4f}{r['dice_control']:>12.4f}"
                  f"{r['site_effect']:>+13.4f}{ve:>15}{pv:>10}")
        print("-" * 86)
        print(f"{'MEAN':<7}{'':>5}{paired['dice_loho'].mean():>12.4f}"
              f"{paired['dice_control'].mean():>12.4f}"
              f"{paired['site_effect'].mean():>+13.4f}"
              f"{paired['volume_effect'].mean():>+15.4f}")
        print(line)

        print("\nHOW TO READ TABLE 2")
        print("  SITE effect   = Dice(LOHO) - Dice(control), same training size.")
        print("                  Strongly negative -> the HOSPITAL caused the drop.")
        print("                  Near zero        -> the hospital did NOT matter.")
        print("  volume effect = Dice(control) - Dice(baseline).")
        print("                  Negative -> smaller training set cost accuracy.")
        if not have_scipy:
            print("\n  (scipy not installed - p-values skipped. Optional: "
                  "pip install scipy)")

    print(f"\nSaved -> {results_dir}/domain_shift_summary.csv")
    print(f"Saved -> {results_dir}/domain_shift_per_patient.csv")
    if paired_rows:
        print(f"Saved -> {results_dir}/domain_shift_paired.csv")


if __name__ == "__main__":
    main()
