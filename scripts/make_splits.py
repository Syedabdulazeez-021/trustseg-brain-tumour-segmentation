"""
make_splits.py
==============
Generates the split CSV files for the multi-institution (domain shift) study.

Run from the project root:

    venv\\Scripts\\python.exe scripts/make_splits.py

WHAT THIS DOES
--------------
Your dataset has 110 patients from 5 hospitals (TCGA tissue source sites),
encoded in the folder name: TCGA_<SITE>_<id>_<date>.

    DU: 45    HT: 34    CS: 16    FG: 14    EZ: 1

This script reads your existing data/split.csv and writes NEW split files
into the splits/ folder. It never modifies your original file and never
touches your images.

FILES PRODUCED
--------------
  splits/split_patient_random.csv  -> honest baseline, random patients
  splits/split_holdout_DU.csv      -> DU is the unseen hospital
  splits/split_holdout_HT.csv      -> HT is the unseen hospital
  splits/split_holdout_CS.csv      -> CS is the unseen hospital
  splits/split_holdout_FG.csv      -> FG is the unseen hospital

EZ has only 1 patient, so it is always kept in training (too small to be a
meaningful test set). State this in your report.
"""

import os
import random

import pandas as pd

SOURCE_CSV = os.path.join("data", "split.csv")
OUT_DIR = "splits"
HOLDOUT_SITES = ["DU", "HT", "CS", "FG"]  # EZ excluded: only 1 patient
VAL_FRACTION = 0.15                        # of the training patients
SEED = 42


def site_of(patient: str) -> str:
    """TCGA_CS_4941_19960909 -> 'CS'"""
    return patient.split("_")[1]


def build_rows(df, val_patients, test_patients):
    """Return a copy of df with the split column rewritten."""
    out = df.copy()

    def label(p):
        if p in test_patients:
            return "test"
        if p in val_patients:
            return "val"
        return "train"

    out["split"] = out["patient"].map(label)
    return out


def summarise(name, out, test_patients):
    n_pat = out.groupby("split")["patient"].nunique()
    n_slice = out["split"].value_counts()
    print(f"\n{name}")
    print(f"  patients -> train {n_pat.get('train', 0)}, "
          f"val {n_pat.get('val', 0)}, test {n_pat.get('test', 0)}")
    print(f"  slices   -> train {n_slice.get('train', 0)}, "
          f"val {n_slice.get('val', 0)}, test {n_slice.get('test', 0)}")
    # Safety check: no patient in more than one split.
    overlap = out.groupby("patient")["split"].nunique().max()
    assert overlap == 1, "LEAK: a patient appears in more than one split!"
    if test_patients:
        sites = {site_of(p) for p in test_patients}
        print(f"  test hospital(s) -> {sorted(sites)}")


def main():
    if not os.path.exists(SOURCE_CSV):
        raise SystemExit(f"Cannot find {SOURCE_CSV}. Run from the project root.")

    df = pd.read_csv(SOURCE_CSV)
    for col in ("image_path", "mask_path", "patient", "split"):
        if col not in df.columns:
            raise SystemExit(f"Expected column '{col}' in {SOURCE_CSV}.")

    patients = sorted(df["patient"].unique())
    print(f"Loaded {len(df)} slices from {len(patients)} patients.")

    by_site = {}
    for p in patients:
        by_site.setdefault(site_of(p), []).append(p)
    print("Hospitals: " + ", ".join(
        f"{s}={len(v)}" for s, v in sorted(by_site.items())))

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # ---------- 1. Honest baseline: random patient split ----------
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n_test = int(round(0.20 * len(shuffled)))
    n_val = int(round(0.15 * len(shuffled)))
    test_p = set(shuffled[:n_test])
    val_p = set(shuffled[n_test:n_test + n_val])

    out = build_rows(df, val_p, test_p)
    path = os.path.join(OUT_DIR, "split_patient_random.csv")
    out.to_csv(path, index=False)
    summarise("split_patient_random.csv  (baseline: mixed hospitals)",
              out, test_p)

    # ---------- 2-5. Leave-one-hospital-out ----------
    for site in HOLDOUT_SITES:
        test_p = set(by_site[site])
        remaining = [p for p in patients if p not in test_p]
        rng2 = random.Random(SEED)
        rng2.shuffle(remaining)
        n_val = max(1, int(round(VAL_FRACTION * len(remaining))))
        val_p = set(remaining[:n_val])

        out = build_rows(df, val_p, test_p)
        path = os.path.join(OUT_DIR, f"split_holdout_{site}.csv")
        out.to_csv(path, index=False)
        summarise(f"split_holdout_{site}.csv  ({site} never seen in training)",
                  out, test_p)

    print(f"\nDone. {1 + len(HOLDOUT_SITES)} files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
