"""
losses.py
=========
The loss function tells the network "how wrong" each prediction is, so the
optimiser knows which direction to nudge the weights.

THE CLASS-IMBALANCE PROBLEM (analogy)
-------------------------------------
In a brain MRI slice, the tumour might be a tiny coin on a large table.
Most pixels are background. If we only used plain Binary Cross-Entropy
(BCE), the network could score "99% pixel accuracy" by lazily predicting
"no tumour anywhere" — useless for us.

OUR FIX: Dice Loss + BCE
------------------------
* Dice Loss directly measures OVERLAP between the predicted blob and the
  true blob. It cares about the shape, not the pixel count, so it is not
  fooled by the huge background.
* BCE gives a smooth, well-behaved gradient for every single pixel, which
  helps training stay stable, especially early on.

Adding them together gives the best of both: BCE keeps training smooth,
Dice forces the network to actually find the tumour.

    total_loss = dice_loss + bce_loss

The model outputs RAW LOGITS, so:
* DiceLoss is created with from_logits=True (it applies sigmoid inside).
* We use BCEWithLogitsLoss (it also applies sigmoid inside, stably).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for binary segmentation.

    Parameters
    ----------
    bce_weight : float
        Multiplier on the BCE term (default 1.0).
    dice_weight : float
        Multiplier on the Dice term (default 1.0).
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # smp DiceLoss(mode="binary") expects logits when from_logits=True.
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model output, shape (N, 1, H, W).
        target : torch.Tensor
            Ground-truth mask in {0,1}, shape (N, 1, H, W), float.
        """
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    # Standalone self-test: needs NO dataset.
    print("=" * 60)
    print("losses.py self-test")
    print("=" * 60)

    criterion = DiceBCELoss()

    # Case 1: a confident-correct prediction should give a LOW loss.
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1.0  # a square "tumour"
    good_logits = torch.where(target > 0.5, 10.0, -10.0)  # very confident & right
    bad_logits = torch.where(target > 0.5, -10.0, 10.0)   # very confident & wrong

    good = criterion(good_logits, target).item()
    bad = criterion(bad_logits, target).item()

    print(f"Loss when prediction is CORRECT : {good:.4f}")
    print(f"Loss when prediction is WRONG   : {bad:.4f}")
    assert good < bad, "A correct prediction must have lower loss than a wrong one!"
    print("\nSUCCESS: combined Dice+BCE loss behaves correctly.")
