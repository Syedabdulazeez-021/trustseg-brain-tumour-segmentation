"""
find_tumor_slices.py
====================
Pulls a handful of REAL MRI slices that are GUARANTEED to contain a tumour
out of your dataset, and copies them to a folder you can upload to the demo.

WHY YOU NEED THIS
-----------------
Most slices in the LGG dataset have NO tumour, so a random upload usually
(correctly) produces an empty result. To verify the app really outlines
tumours, you must feed it a slice that actually has one. This script finds
those slices by checking which masks are non-empty.

USAGE (from the project root):
    python scripts/find_tumor_slices.py                 # uses data/split.csv, test split
    python scripts/find_tumor_slices.py --n 8           # copy 8 slices
    python scripts/find_tumor_slices.py --split val     # pick from a different split
    python scripts/find_tumor_slices.py --raw_root kaggle_3m   # scan raw data instead

Output: a folder `verify_samples/` containing:
    tumour_01.tif ...   (upload THESE to the Streamlit app)
    tumour_01_mask.tif  (the ground-truth outline, for your own comparison)
"""

import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd


def mask_has_tumour(mask_path: str, min_pixels: int = 30) -> int:
    """Return the tumour pixel count for a mask file (0 if empty/unreadable)."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0
    return int((m > 127).sum())


def from_split_csv(csv_path: str, split: str):
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split] if "split" in df.columns else df
    return list(zip(df["image_path"], df["mask_path"]))


def from_raw_root(raw_root: str):
    import glob
    pairs = []
    for mask_path in glob.glob(os.path.join(raw_root, "*", "*_mask.tif")):
        image_path = mask_path.replace("_mask.tif", ".tif")
        if os.path.exists(image_path):
            pairs.append((image_path, mask_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Extract tumour-containing slices.")
    ap.add_argument("--split_csv", default="data/split.csv")
    ap.add_argument("--split", default="test",
                    help="Which split to draw from (train/val/test).")
    ap.add_argument("--raw_root", default=None,
                    help="Scan raw dataset instead of using split.csv.")
    ap.add_argument("--n", type=int, default=5, help="How many slices to copy.")
    ap.add_argument("--out_dir", default="verify_samples")
    args = ap.parse_args()

    # Gather candidate (image, mask) pairs.
    if args.raw_root:
        pairs = from_raw_root(args.raw_root)
        source = args.raw_root
    else:
        if not os.path.exists(args.split_csv):
            raise FileNotFoundError(
                f"{args.split_csv} not found. Run data/prepare_data.py first, "
                "or pass --raw_root kaggle_3m to scan the raw dataset."
            )
        pairs = from_split_csv(args.split_csv, args.split)
        source = f"{args.split_csv} (split={args.split})"

    print(f"Scanning {len(pairs)} slices from {source} for tumours...")

    # Rank by tumour size (biggest, clearest tumours first).
    scored = []
    for img_p, msk_p in pairs:
        px = mask_has_tumour(msk_p)
        if px > 30:
            scored.append((px, img_p, msk_p))
    scored.sort(reverse=True)

    if not scored:
        print("No tumour-containing slices found. Check your dataset path.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.n, len(scored))
    print(f"Found {len(scored)} tumour slices; copying the {n} largest.\n")

    for i, (px, img_p, msk_p) in enumerate(scored[:n], start=1):
        dst_img = os.path.join(args.out_dir, f"tumour_{i:02d}.tif")
        dst_msk = os.path.join(args.out_dir, f"tumour_{i:02d}_mask.tif")
        shutil.copy(img_p, dst_img)
        shutil.copy(msk_p, dst_msk)
        print(f"  {dst_img}   (tumour area: {px} px)")

    print(f"\nDone. Upload the tumour_XX.tif files (NOT the _mask ones) to the "
          f"Streamlit app.\nThe app should outline the tumour in red and the "
          f"heatmap should light up.")


if __name__ == "__main__":
    main()
