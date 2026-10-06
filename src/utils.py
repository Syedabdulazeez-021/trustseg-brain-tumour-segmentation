"""
utils.py
========
Small helper functions shared across the project: choosing the device,
de-normalising an image for display, drawing a tumour overlay, and saving
training curves.

THE OVERLAY (analogy)
---------------------
The model gives us a black/white mask. On its own that is hard to judge.
An overlay paints the predicted tumour as a translucent red wash directly
ON TOP of the grey MRI — exactly how a radiologist would mark a printout
with a highlighter — so a human can instantly see whether it looks right.
"""

import os

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(image_t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> uint8 RGB image for display.

    Parameters
    ----------
    image_t : torch.Tensor
        A (3, H, W) normalised tensor as produced by the Dataset.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB image.
    """
    img = image_t.detach().cpu().numpy().transpose(1, 2, 0)  # HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1) * 255.0
    return img.astype(np.uint8)


def make_overlay(rgb_image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4, color=(255, 0, 0)) -> np.ndarray:
    """Paint `mask` over `rgb_image` as a translucent coloured wash.

    Parameters
    ----------
    rgb_image : np.ndarray
        (H, W, 3) uint8 base image.
    mask : np.ndarray
        (H, W) binary array in {0,1} (or {0,255}).
    alpha : float
        Opacity of the overlay colour (0 transparent .. 1 solid).
    color : tuple
        RGB colour of the overlay (default red).

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 overlay image.
    """
    base = rgb_image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    mask_bool = mask > 0.5
    color_layer = np.zeros_like(base)
    color_layer[:] = color
    out = base.copy()
    out[mask_bool] = (
        alpha * color_layer[mask_bool] + (1 - alpha) * base[mask_bool]
    ).astype(np.uint8)
    return out


def save_training_curves(history: dict, out_path: str) -> None:
    """Plot loss and Dice over epochs and save to `out_path`.

    `history` must contain lists: train_loss, val_loss, train_dice, val_dice.
    Imported lazily so importing utils does not require matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], label="train")
    axes[1].plot(epochs, history["val_dice"], label="val")
    axes[1].set_title("Dice coefficient over epochs")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_prediction_panel(rgb_image: np.ndarray, true_mask: np.ndarray,
                          pred_mask: np.ndarray, out_path: str) -> None:
    """Save a 3-panel figure: original | ground truth | prediction overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_overlay = make_overlay(rgb_image, true_mask, color=(0, 255, 0))
    pred_overlay = make_overlay(rgb_image, pred_mask, color=(255, 0, 0))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(rgb_image)
    axes[0].set_title("MRI slice")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground truth (green)")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction (red)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    # Standalone self-test: needs NO dataset.
    print("=" * 60)
    print("utils.py self-test")
    print("=" * 60)

    print("Device:", get_device())

    # Fake normalised image -> denormalise -> overlay.
    fake = torch.randn(3, 64, 64)
    rgb = denormalize(fake)
    print("denormalize -> shape", rgb.shape, "dtype", rgb.dtype)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8

    mask = np.zeros((64, 64), np.float32)
    mask[20:40, 20:40] = 1.0
    over = make_overlay(rgb, mask)
    print("overlay -> shape", over.shape)
    assert over.shape == (64, 64, 3)

    # Curves with 3 fake epochs.
    hist = {
        "train_loss": [0.9, 0.6, 0.4], "val_loss": [1.0, 0.7, 0.5],
        "train_dice": [0.3, 0.6, 0.75], "val_dice": [0.25, 0.55, 0.7],
    }
    save_training_curves(hist, "results/_selftest_curves.png")
    print("Saved results/_selftest_curves.png")
    assert os.path.exists("results/_selftest_curves.png")

    print("\nSUCCESS: utils helpers all work.")
