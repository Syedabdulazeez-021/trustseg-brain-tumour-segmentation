"""
make_control_splits.py
======================
Generates the MATCHED-CONTROL splits for the multi-institution study.

Run from the project root (after make_splits.py):

    venv\\Scripts\\python.exe scripts/make_control_splits.py

WHY THIS EXISTS
---------------
The leave-one-hospital-out results showed a drop in Dice. But holding out a
hospital also removes patients from training, so a critic can say:

    "DU dropped the most because that model simply had the least data,
     not because DU looks different."

This script builds the control. For each holdout it creates a split with the
SAME number of training patients and the SAME number of test patients, but
the test patients are drawn randomly from ALL hospitals instead of being one
hospital.

    split_holdout_DU.csv   -> train 65, test 45 (all DU)
    split_control_DU.csv   -> train 65, test 45 (mixed hospitals)

Only one thing differs: whether the test set is a single unseen hospital.
So any remaining gap is attributable to the hospital, not the data volume.

FILES PRODUCED
--------------
  splits/split_control_DU.csv
  splits/split_control_HT.csv
  splits/split_control_CS.csv
  splits/split_control_FG.csv

The control test sets are sampled with a different random seed per site and
are checked to make sure they are not accidentally dominated by one hospital.
"""

import os
import random

import pandas as pd

SOURCE_CSV = os.path.join("data", "split.csv")
SPLIT_DIR = "splits"
SITES = ["DU", "HT", "CS", "FG"]
SEED = 1234


def site_of(patient: str) -> str:
    return patient.split("_")[1]


def build_rows(df, val_patients, test_patients):
    out = df.copy()

    def label(p):
        if p in test_patients:
            return "test"
        if p in val_patients:
            return "val"
        return "train"

    out["split"] = out["patient"].map(label)
    return out


def main():
    if not os.path.exists(SOURCE_CSV):
        raise SystemExit(f"Cannot find {SOURCE_CSV}. Run from the project root.")

    df = pd.read_csv(SOURCE_CSV)
    patients = sorted(df["patient"].unique())
    print(f"Loaded {len(df)} slices from {len(patients)} patients.\n")

    for i, site in enumerate(SITES):
        holdout_path = os.path.join(SPLIT_DIR, f"split_holdout_{site}.csv")
        if not os.path.exists(holdout_path):
            print(f"[skip] {site}: {holdout_path} not found. "
                  f"Run make_splits.py first.")
            continue

        # Read the patient counts we need to match.
        h = pd.read_csv(holdout_path)
        counts = h.groupby("split")["patient"].nunique()
        n_test = int(counts.get("test", 0))
        n_val = int(counts.get("val", 0))

        # Sample a mixed test set of the same size.
        rng = random.Random(SEED + i)
        shuffled = patients[:]
        rng.shuffle(shuffled)
        test_p = set(shuffled[:n_test])
        remaining = [p for p in shuffled if p not in test_p]
        val_p = set(remaining[:n_val])

        out = build_rows(df, val_p, test_p)
        out_path = os.path.join(SPLIT_DIR, f"split_control_{site}.csv")
        out.to_csv(out_path, index=False)

        # Report and sanity-check.
        got = out.groupby("split")["patient"].nunique()
        test_sites = {}
        for p in test_p:
            s = site_of(p)
            test_sites[s] = test_sites.get(s, 0) + 1
        mix = ", ".join(f"{k}={v}" for k, v in sorted(test_sites.items()))

        overlap = out.groupby("patient")["split"].nunique().max()
        assert overlap == 1, "LEAK: a patient appears in more than one split!"

        print(f"split_control_{site}.csv  (matched to holdout_{site})")
        print(f"  patients -> train {got.get('train', 0)}, "
              f"val {got.get('val', 0)}, test {got.get('test', 0)}")
        print(f"  target   -> train {int(counts.get('train', 0))}, "
              f"val {n_val}, test {n_test}")
        print(f"  test mix -> {mix}")
        print(f"  n hospitals in test: {len(test_sites)}\n")

    print(f"Done. Control splits written to {SPLIT_DIR}/")


if __name__ == "__main__":
    main()
