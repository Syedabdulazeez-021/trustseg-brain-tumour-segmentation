"""
uncertainty.py
==============
Measures HOW CONFIDENT the segmentation model is, per slice, and tests
whether that confidence actually predicts when the model is wrong.

Run from the project root (inference only - takes a few minutes, no training):

    set KMP_DUPLICATE_LIB_OK=TRUE
    venv\\Scripts\\python.exe scripts/uncertainty.py

    # or point it at a different model / split:
    venv\\Scripts\\python.exe scripts/uncertainty.py --tag DU

WHY TEST-TIME AUGMENTATION, NOT MC DROPOUT
------------------------------------------
MC Dropout needs dropout layers switched on at inference. `smp.Unet` is built
without any dropout at all, so there is nothing to sample from - MC Dropout
simply cannot run on these checkpoints.

Test-Time Augmentation (TTA) instead feeds the SAME slice through the model
in 8 different orientations (the D4 symmetry group: 4 rotations x optional
mirror), un-rotates each prediction, and compares them. Those 8 views should
agree. Where they disagree, the model's decision is unstable - that is the
uncertainty signal.

This is a particularly honest choice here, because the model was TRAINED with
flips and 90-degree rotations. It has every reason to be invariant to them,
so disagreement is genuine model uncertainty rather than an artefact of
showing it something it never saw.

FOUR UNCERTAINTY SCORES (we compute all four and report which works best)
------------------------------------------------------------------------
  mean_entropy   average pixel-wise entropy of the averaged probability map
  fg_entropy     total entropy divided by predicted tumour area (normalised,
                 so a big confident tumour is not penalised for being big)
  tta_disagree   1 - (mean pairwise Dice between the 8 binarised TTA views).
                 Directly interpretable: "how much do the views disagree?"
  boundary_ent   mean entropy over genuinely ambiguous pixels only
                 (those with averaged probability between 0.05 and 0.95)

THE HEADLINE EXPERIMENT - THE RETENTION CURVE
---------------------------------------------
Sort every slice from most-confident to least-confident. Then imagine handing
the least-confident X% to a radiologist and letting the model keep the rest.
Plot the Dice of what the model kept, as X grows.

If the uncertainty score is meaningful, that curve RISES: the model is
discarding exactly the cases it was going to get wrong. That is a concrete,
deployable safety feature - automatic quality control - and it is measured,
not asserted.

OUTPUTS
-------
  results/uncertainty_<tag>_per_slice.csv   one row per test slice
  results/uncertainty_<tag>_retention.png   the headline figure
  results/uncertainty_<tag>_examples.png    most/least confident cases
"""

import argparse
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
from src.utils import denormalize, get_device, make_overlay
from train import load_config

THRESHOLD = 0.5
EPS = 1e-7

# The 8 elements of D4: (number of 90-degree rotations, mirror first?)
TTA_OPS = [(k, f) for f in (False, True) for k in (0, 1, 2, 3)]


# ----------------------------------------------------------------------
# TTA machinery
# ----------------------------------------------------------------------
def tta_forward(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply mirror-then-rotate to a (B, C, H, W) tensor."""
    if flip:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k, dims=[2, 3])


def tta_inverse(y: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Exactly undo `tta_forward`."""
    y = torch.rot90(y, -k, dims=[2, 3])
    if flip:
        y = torch.flip(y, dims=[3])
    return y


@torch.no_grad()
def tta_probabilities(model, images: torch.Tensor) -> torch.Tensor:
    """Return (T, B, 1, H, W) probability maps, all mapped back to the
    original orientation."""
    outs = []
    for k, flip in TTA_OPS:
        xt = tta_forward(images, k, flip)
        logits = model(xt)
        probs = torch.sigmoid(logits)
        outs.append(tta_inverse(probs, k, flip))
    return torch.stack(outs, dim=0)


# ----------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------
def dice_np(pred: np.ndarray, target: np.ndarray) -> float:
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + EPS) / (denom + EPS)


def pairwise_disagreement(binary_stack: np.ndarray) -> float:
    """1 - mean pairwise Dice between the T binarised TTA views.

    0.0 = all views identical (confident). Higher = less stable.
    """
    T = binary_stack.shape[0]
    scores = []
    for i in range(T):
        for j in range(i + 1, T):
            scores.append(dice_np(binary_stack[i], binary_stack[j]))
    return float(1.0 - np.mean(scores)) if scores else 0.0


def entropy_map(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def spearman(a, b):
    """Spearman rho without requiring scipy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------
# Main pass
# ----------------------------------------------------------------------
@torch.no_grad()
def run(model, split_csv, image_size, device, batch_size, keep_examples=6):
    ds = BrainMRIDataset(split_csv, "test", image_size)
    rows = pd.read_csv(split_csv)
    rows = rows[rows["split"] == "test"].reset_index(drop=True)
    if len(rows) != len(ds):
        raise SystemExit(
            f"Row mismatch: CSV has {len(rows)} test rows, dataset built "
            f"{len(ds)}. Cannot align patients.")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    recs, examples = [], []
    idx = 0
    for images, masks in tqdm(loader, leave=False, desc="  TTA passes"):
        images = images.to(device)
        stack = tta_probabilities(model, images)          # (T,B,1,H,W)
        mean_p = stack.mean(dim=0)                        # (B,1,H,W)

        stack_np = stack.cpu().numpy()
        mean_np = mean_p.cpu().numpy()
        gts = masks.numpy()
        imgs_cpu = images.cpu()

        for b in range(mean_np.shape[0]):
            p = mean_np[b, 0]
            g = (gts[b, 0] > 0.5).astype(np.float32)
            pred = (p > THRESHOLD).astype(np.float32)
            views = (stack_np[:, b, 0] > THRESHOLD).astype(np.float32)

            ent = entropy_map(p)
            amb = (p > 0.05) & (p < 0.95)

            recs.append({
                "patient": rows.loc[idx, "patient"],
                "image_path": rows.loc[idx, "image_path"],
                "dice": dice_np(pred, g),
                "has_tumour": bool(g.sum() > 0),
                "pred_area": float(pred.sum()),
                "true_area": float(g.sum()),
                "mean_entropy": float(ent.mean()),
                "fg_entropy": float(ent.sum() / (pred.sum() + 1.0)),
                "tta_disagree": pairwise_disagreement(views),
                "boundary_ent": float(ent[amb].mean()) if amb.any() else 0.0,
            })

            if len(examples) < 200:
                examples.append((idx, imgs_cpu[b], g, p, pred))
            idx += 1

    return pd.DataFrame(recs), examples


def retention_table(df, score_col, steps=(0, 10, 20, 30, 40, 50)):
    """Dice of retained slices as the most-uncertain X% are referred away."""
    d = df.sort_values(score_col, ascending=True).reset_index(drop=True)
    n = len(d)
    out = []
    for pct in steps:
        keep = max(1, int(round(n * (1 - pct / 100.0))))
        out.append({
            "referred_pct": pct,
            "n_kept": keep,
            "dice_kept": float(d.loc[:keep - 1, "dice"].mean()),
        })
    return pd.DataFrame(out)


def plot_retention(tables, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, tb in tables.items():
        ax.plot(tb["referred_pct"], tb["dice_kept"], marker="o", label=name)
    ax.set_xlabel("% of most-uncertain slices referred to a radiologist")
    ax.set_ylabel("Dice on the slices the model kept")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="uncertainty score", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_examples(df, examples, score_col, out_path):
    """Three most-confident and three least-confident tumour slices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {i: e for i, *e in [(e[0], *e[1:]) for e in examples]}
    tum = df[df["has_tumour"]].copy()
    tum["row"] = tum.index
    avail = tum[tum["row"].isin(lookup.keys())]
    if len(avail) < 2:
        return False

    low = avail.nsmallest(3, score_col)
    high = avail.nlargest(3, score_col)
    picks = [("CONFIDENT", r) for _, r in low.iterrows()] + \
            [("UNCERTAIN", r) for _, r in high.iterrows()]

    fig, axes = plt.subplots(len(picks), 3, figsize=(9.5, 3.1 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])

    for r, (label, rec) in enumerate(picks):
        img_t, g, p, pred = lookup[int(rec["row"])]
        rgb = denormalize(img_t)
        axes[r, 0].imshow(make_overlay(rgb, g, color=(0, 255, 0)))
        axes[r, 0].set_title(f"{label} - ground truth", fontsize=10)
        axes[r, 1].imshow(make_overlay(rgb, pred, color=(255, 0, 0)))
        axes[r, 1].set_title(f"prediction (Dice {rec['dice']:.3f})", fontsize=10)
        im = axes[r, 2].imshow(entropy_map(p), cmap="inferno")
        axes[r, 2].set_title(f"uncertainty ({score_col} {rec[score_col]:.3f})",
                             fontsize=10)
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046)
        for c in range(3):
            axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="TTA uncertainty analysis.")
    ap.add_argument("--tag", default="random",
                    help="Model tag, e.g. random, DU, HT, CS, FG.")
    ap.add_argument("--split_csv", default=None,
                    help="Defaults to the split matching --tag.")
    ap.add_argument("--model_path", default=None,
                    help="Defaults to results/best_model_<tag>.pth")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    root, ext = os.path.splitext(cfg["train"]["best_model_path"])
    model_path = args.model_path or f"{root}_{args.tag}{ext}"
    if args.split_csv:
        split_csv = args.split_csv
    elif args.tag == "random":
        split_csv = "splits/split_patient_random.csv"
    else:
        split_csv = f"splits/split_holdout_{args.tag}.csv"

    for p in (model_path, split_csv):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}")

    device = get_device()
    results_dir = cfg["train"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Model      : {model_path}")
    print(f"Split      : {split_csv}")
    print(f"TTA views  : {len(TTA_OPS)} (D4 symmetry group)\n")

    model = load_trained_model(model_path, device=device)
    df, examples = run(model, split_csv, cfg["data"]["image_size"],
                       device, args.batch_size)

    csv_path = os.path.join(results_dir, f"uncertainty_{args.tag}_per_slice.csv")
    df.to_csv(csv_path, index=False)

    tum = df[df["has_tumour"]].copy()
    scores = ["tta_disagree", "fg_entropy", "mean_entropy", "boundary_ent"]

    line = "=" * 72
    print("\n" + line)
    print(f"UNCERTAINTY ANALYSIS - model '{args.tag}'")
    print(line)
    print(f"Test slices            : {len(df)}  "
          f"({len(tum)} contain tumour)")
    print(f"Dice (all slices)      : {df['dice'].mean():.4f}")
    print(f"Dice (tumour only)     : {tum['dice'].mean():.4f}")

    print("\nDoes uncertainty predict error?  (Spearman rho vs Dice;")
    print("strongly NEGATIVE = more uncertainty really does mean worse Dice)")
    print("-" * 72)
    print(f"{'score':<16}{'rho (tumour slices)':>22}{'rho (all slices)':>20}")
    print("-" * 72)
    rhos = {}
    for sc in scores:
        r_t = spearman(tum[sc], tum["dice"])
        r_a = spearman(df[sc], df["dice"])
        rhos[sc] = r_t
        print(f"{sc:<16}{r_t:>22.4f}{r_a:>20.4f}")
    print("-" * 72)

    best = min(rhos, key=lambda k: (rhos[k] if np.isfinite(rhos[k]) else 1e9))
    print(f"Strongest signal: {best}  (rho = {rhos[best]:.4f})")

    # ---- Retention curve: the headline result ----
    tables = {sc: retention_table(tum, sc) for sc in scores}
    tb = tables[best]
    print("\n" + line)
    print("RETENTION CURVE - automatic quality control")
    print(line)
    print(f"Using '{best}'. Slices are ranked by uncertainty; the most")
    print("uncertain are referred to a radiologist and the model keeps the rest.")
    print("-" * 72)
    print(f"{'% referred':>12}{'slices kept':>14}{'Dice on kept':>16}{'gain':>10}")
    print("-" * 72)
    base = tb.loc[0, "dice_kept"]
    for _, r in tb.iterrows():
        print(f"{int(r['referred_pct']):>12}{int(r['n_kept']):>14}"
              f"{r['dice_kept']:>16.4f}{r['dice_kept'] - base:>+10.4f}")
    print(line)

    fig_path = os.path.join(results_dir, f"uncertainty_{args.tag}_retention.png")
    plot_retention(tables, fig_path,
                   f"Selective prediction - model '{args.tag}' "
                   f"(tumour-bearing slices)")

    ex_path = os.path.join(results_dir, f"uncertainty_{args.tag}_examples.png")
    made = plot_examples(df, examples, best, ex_path)

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {fig_path}")
    if made:
        print(f"Saved -> {ex_path}")

    if np.isfinite(rhos[best]) and rhos[best] < -0.25:
        print("\nRESULT: uncertainty is informative - the model can flag its "
              "own likely failures.")
    else:
        print("\nRESULT: the correlation is weak. Report this honestly; it "
              "means TTA uncertainty does not track error on this model.")


if __name__ == "__main__":
    main()
