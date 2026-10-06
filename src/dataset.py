"""
dataset.py
==========
Turns image files on disk into batches of tensors the network can eat.

WHAT A pytorch Dataset DOES (analogy)
-------------------------------------
A Dataset is like a librarian: you ask for "item number 37" and it fetches
that MRI slice and its tumour mask, cleans them up (resize, normalise,
maybe flip for augmentation) and hands them back as tensors. The DataLoader
then stacks many items into a batch.

KEY CORRECTNESS POINTS
----------------------
1. Masks are binarised to {0, 1}. The raw .tif masks store 0 or 255; if we
   forgot to divide, the loss/metrics would be nonsense.
2. The SAME geometric augmentation is applied to image AND mask. Albumentations
   does this automatically when we pass mask=mask, so a flipped brain still
   lines up with its flipped tumour outline.
3. EMPTY masks (slices with no tumour) are valid and common — they must load
   without error and contribute background-only supervision.
4. Images are normalised with ImageNet mean/std because the pretrained
   resnet34 encoder was trained expecting exactly that scaling.
"""

import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ImageNet statistics the pretrained encoder expects.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size: int = 256, train: bool = False) -> A.Compose:
    """Build the albumentations pipeline.

    Training adds random flips/rotations/brightness so the model sees more
    variety and generalises better. Validation/test only resize + normalise.
    """
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _read_rgb(path: str) -> np.ndarray:
    """Read an image (.tif or .png) as an RGB uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: str) -> np.ndarray:
    """Read a mask as a single-channel uint8 array (0 or 255)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


class BrainMRIDataset(Dataset):
    """Dataset of (MRI slice, tumour mask) pairs from a split CSV.

    The CSV must have at least the columns: image_path, mask_path, split.
    """

    def __init__(self, csv_path: str, split: str, image_size: int = 256):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"No rows for split='{split}' in {csv_path}. "
                "Did you run data/prepare_data.py?"
            )
        self.transform = get_transforms(image_size, train=(split == "train"))
        self.split = split

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = _read_rgb(row["image_path"])
        mask = _read_mask(row["mask_path"])

        # Binarise the mask to {0,1}. Empty masks stay all-zeros (valid).
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]                 # (3, H, W) float
        mask_t = augmented["mask"].unsqueeze(0).float()  # (1, H, W) float

        return image_t, mask_t


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Standalone self-test: needs NO real dataset. We synthesise two tiny
    # fake slices (one WITH a tumour, one EMPTY) on the fly, write a CSV,
    # and confirm the Dataset returns correctly shaped, binarised tensors.
    # ------------------------------------------------------------------
    import tempfile

    print("=" * 60)
    print("dataset.py self-test (synthetic data, no download needed)")
    print("=" * 60)

    tmp = tempfile.mkdtemp()
    rows = []

    # Slice 1: has a tumour (white square in the mask).
    img1 = (np.random.rand(120, 120, 3) * 255).astype(np.uint8)
    m1 = np.zeros((120, 120), np.uint8)
    m1[40:80, 40:80] = 255
    cv2.imwrite(os.path.join(tmp, "p1_1.png"), img1)
    cv2.imwrite(os.path.join(tmp, "p1_1_mask.png"), m1)
    rows.append({"image_path": os.path.join(tmp, "p1_1.png"),
                 "mask_path": os.path.join(tmp, "p1_1_mask.png"),
                 "patient": "p1", "split": "train"})

    # Slice 2: EMPTY mask (no tumour) — must still load fine.
    img2 = (np.random.rand(120, 120, 3) * 255).astype(np.uint8)
    m2 = np.zeros((120, 120), np.uint8)
    cv2.imwrite(os.path.join(tmp, "p2_1.png"), img2)
    cv2.imwrite(os.path.join(tmp, "p2_1_mask.png"), m2)
    rows.append({"image_path": os.path.join(tmp, "p2_1.png"),
                 "mask_path": os.path.join(tmp, "p2_1_mask.png"),
                 "patient": "p2", "split": "train"})

    csv_path = os.path.join(tmp, "split.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ds = BrainMRIDataset(csv_path, split="train", image_size=128)
    print(f"Dataset length: {len(ds)}")
    for i in range(len(ds)):
        img_t, msk_t = ds[i]
        uniq = torch.unique(msk_t).tolist()
        print(f"  item {i}: image {tuple(img_t.shape)}, mask {tuple(msk_t.shape)}, "
              f"mask values {uniq}")
        assert img_t.shape == (3, 128, 128)
        assert msk_t.shape == (1, 128, 128)
        # Mask must be binary {0} or {0,1}.
        assert set(uniq).issubset({0.0, 1.0})

    print("\nSUCCESS: dataset loads, resizes, binarises, and handles empty masks.")
