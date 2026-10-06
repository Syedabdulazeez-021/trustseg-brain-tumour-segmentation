"""
metrics.py
==========
Numbers that tell us how GOOD the segmentation is (the loss is for training;
these are for reporting and for choosing the best model).

QUICK DICTIONARY (analogy: a tumour "treasure hunt")
----------------------------------------------------
For each pixel the model either says "tumour" or "not tumour", and it is
either right or wrong. Four buckets:
  TP = true positives  : tumour pixels we correctly found
  FP = false positives : background we wrongly called tumour (false alarm)
  FN = false negatives : tumour we missed
  TN = true negatives  : background we correctly ignored

* Dice  = 2*TP / (2*TP + FP + FN)
          Overlap score between prediction and truth. 1.0 = perfect.
          This is our PRIMARY metric (target >= 0.80).
* IoU   = TP / (TP + FP + FN)
          "Intersection over Union" (Jaccard). Stricter cousin of Dice.
* Sensitivity (recall) = TP / (TP + FN)
          Of all real tumour pixels, how many did we catch? (Missing tumour
          is dangerous, so we watch this closely.)
* Specificity = TN / (TN + FP)
          Of all healthy pixels, how many did we correctly leave alone?

All functions take RAW LOGITS plus a threshold, apply sigmoid internally,
and return a single float averaged over the batch.
"""

import torch

_EPS = 1e-7  # avoids divide-by-zero on empty masks


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Sigmoid the logits then threshold to a {0,1} float tensor."""
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> float:
    """Mean Dice over the batch."""
    pred = _binarize(logits, threshold)
    target = (target > 0.5).float()
    dims = (1, 2, 3)  # per-image, keep batch dim
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + _EPS) / (denom + _EPS)
    return dice.mean().item()


def iou_score(logits: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5) -> float:
    """Mean Intersection-over-Union (Jaccard) over the batch."""
    pred = _binarize(logits, threshold)
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - inter
    iou = (inter + _EPS) / (union + _EPS)
    return iou.mean().item()


def sensitivity(logits: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """Mean sensitivity / recall on tumour pixels = TP / (TP + FN)."""
    pred = _binarize(logits, threshold)
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    sens = (tp + _EPS) / (tp + fn + _EPS)
    return sens.mean().item()


def specificity(logits: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """Mean specificity on background pixels = TN / (TN + FP)."""
    pred = _binarize(logits, threshold)
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    tn = ((1 - pred) * (1 - target)).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    spec = (tn + _EPS) / (tn + fp + _EPS)
    return spec.mean().item()


def all_metrics(logits: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> dict:
    """Convenience: return every metric in one dict."""
    return {
        "dice": dice_coefficient(logits, target, threshold),
        "iou": iou_score(logits, target, threshold),
        "sensitivity": sensitivity(logits, target, threshold),
        "specificity": specificity(logits, target, threshold),
    }


if __name__ == "__main__":
    # Standalone self-test: needs NO dataset.
    print("=" * 60)
    print("metrics.py self-test")
    print("=" * 60)

    target = torch.zeros(4, 1, 64, 64)
    target[:, :, 16:48, 16:48] = 1.0  # a square tumour

    # A PERFECT prediction (logits strongly match the target).
    perfect = torch.where(target > 0.5, 10.0, -10.0)
    m = all_metrics(perfect, target)
    print("Perfect prediction:", {k: round(v, 3) for k, v in m.items()})
    assert m["dice"] > 0.99 and m["iou"] > 0.99

    # A prediction that misses HALF the tumour -> sensitivity should drop.
    half = perfect.clone()
    half[:, :, 32:48, :] = -10.0  # erase bottom half of the tumour
    m2 = all_metrics(half, target)
    print("Half-missed prediction:", {k: round(v, 3) for k, v in m2.items()})
    assert m2["sensitivity"] < m["sensitivity"]

    print("\nSUCCESS: all metrics compute and behave sensibly.")
