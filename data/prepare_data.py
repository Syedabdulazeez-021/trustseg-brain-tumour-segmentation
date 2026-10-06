"""
prepare_data.py
===============
Scans the downloaded LGG MRI dataset, pairs every MRI slice with its tumour
mask, and splits the data into train / val / test — splitting BY PATIENT.

WHY PATIENT-LEVEL SPLITTING MATTERS (the most important idea here)
-----------------------------------------------------------------
Each patient contributes ~20-30 nearly-identical neighbouring slices. If we
shuffled all images and split randomly, slice #14 of a patient could land in
"train" while slice #15 of the SAME patient lands in "test". The model would
then be tested on brains it has basically already seen — scores look amazing
but are a lie. By keeping every slice of one patient inside ONE split, the
test set contains only brains the model has never encountered, giving an
honest estimate of real-world performance.

OUTPUT
------
A CSV (default data/split.csv) with columns:
    image_path, mask_path, patient, split
Empty-mask slices are KEPT (they are valid background-only examples).
"""

import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def find_pairs(raw_root: str):
    """Return a list of dicts pairing each slice with its mask + patient id.

    Expected layout:
        raw_root/
            TCGA_XX_..../
                TCGA_XX_..._1.tif
                TCGA_XX_..._1_mask.tif
                ...
    """
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(
            f"Dataset folder not found: {raw_root}\n"
            "Download the LGG MRI Segmentation dataset from Kaggle and extract "
            "it so that this folder contains the TCGA_* patient sub-folders.\n"
            "  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"
        )

    pairs = []
    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(raw_root, "*")) if os.path.isdir(d)
    )
    for pdir in patient_dirs:
        patient = os.path.basename(pdir)
        # Masks end with "_mask.tif"; images are the rest.
        masks = glob.glob(os.path.join(pdir, "*_mask.tif"))
        for mask_path in masks:
            image_path = mask_path.replace("_mask.tif", ".tif")
            if os.path.exists(image_path):
                pairs.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "patient": patient,
                })
    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found under {raw_root}. "
            "Check that the dataset extracted correctly."
        )
    return pairs


def split_patients(patients, n_val, n_test, seed):
    """Split a list of unique patient ids into train/val/test sets."""
    patients = sorted(set(patients))
    # First peel off the test patients, then the val patients.
    train_val, test = train_test_split(
        patients, test_size=n_test, random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=n_val, random_state=seed
    )
    return set(train), set(val), set(test)


def build_split_csv(raw_root, out_csv, n_val=15, n_test=15, seed=42):
    """Create the split CSV and return the DataFrame."""
    print(f"Scanning dataset under: {raw_root}")
    pairs = find_pairs(raw_root)
    df = pd.DataFrame(pairs)
    n_patients = df["patient"].nunique()
    print(f"Found {len(df)} slices from {n_patients} patients.")

    train_p, val_p, test_p = split_patients(
        df["patient"].tolist(), n_val, n_test, seed
    )

    def assign(p):
        if p in train_p:
            return "train"
        if p in val_p:
            return "val"
        return "test"

    df["split"] = df["patient"].apply(assign)

    # --- Safety check: no patient may appear in two splits ---
    overlap = (
        (train_p & val_p) | (train_p & test_p) | (val_p & test_p)
    )
    assert not overlap, f"LEAK! Patients in multiple splits: {overlap}"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\nSplit summary (by patient -> by slice):")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        print(f"  {split:5s}: {sub['patient'].nunique():3d} patients, "
              f"{len(sub):5d} slices")
    print(f"\nSaved -> {out_csv}")
    print("Patient-level split verified: no patient appears in two splits.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build patient-level split CSV.")
    parser.add_argument(
        "--raw_root", default="lgg-mri-segmentation/kaggle_3m",
        help="Folder containing the TCGA_* patient sub-folders.",
    )
    parser.add_argument("--out_csv", default="data/split.csv")
    parser.add_argument("--n_val", type=int, default=15)
    parser.add_argument("--n_test", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_split_csv(
        args.raw_root, args.out_csv, args.n_val, args.n_test, args.seed
    )
