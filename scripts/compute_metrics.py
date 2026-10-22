"""
compute_metrics.py
==================
Computes the classification-style metrics the evaluation rubric asks for -
precision, recall, F1, specificity, accuracy - plus two confusion matrices,
for the baseline model on the held-out test set.

Run from the PROJECT ROOT:

    set KMP_DUPLICATE_LIB_OK=TRUE
    venv\\Scripts\\python.exe scripts\\compute_metrics.py

    # or for a different model:
    venv\\Scripts\\python.exe scripts\\compute_metrics.py --tag DU

WHY THIS SCRIPT EXISTS
----------------------
Segmentation is normally reported with Dice and IoU. The rubric asks for
precision, recall and F1. These are not a different measurement - they are
the same confusion-matrix quantities, aggregated differently, and for binary
segmentation:

        F1  =  2PR / (P + R)  =  2TP / (2TP + FP + FN)  =  Dice

F1 and Dice are algebraically identical. Computing them separately is still
worth doing, because precision and recall DECOMPOSE the Dice score and tell
you something Dice alone hides: whether the model's errors are
over-segmentation (low precision, it paints outside the tumour) or
under-segmentation (low recall, it misses tumour). Reporting all three shows
the examiner you understand what Dice is actually made of.

TWO WAYS TO AGGREGATE - BOTH ARE REPORTED
-----------------------------------------
  MICRO (global)  pool every pixel from every slice into one confusion
                  matrix, then compute the metrics once. Large tumours
                  dominate, because they contribute more pixels.

  MACRO (per-slice) compute the metrics on each slice separately, then
                  average. Every slice counts equally, so a small tumour
                  matters as much as a large one. The "mean Dice 0.8176"
                  quoted throughout this project is a macro average.

They answer different questions and they do not agree. Knowing why is the
kind of thing a viva asks about.

TWO CONFUSION MATRICES
----------------------
  1. PIXEL-LEVEL    every pixel classified as tumour / not tumour. Honest,
                    but dominated by true negatives: most of a brain slice
                    is background, so "accuracy" looks absurdly high and
                    means almost nothing.

  2. SLICE-LEVEL    did the model detect ANY tumour on a slice that has one?
                    This is the detection question, and it is where the
                    project's 5.0% empty-prediction finding lives.

OUTPUTS
-------
  results/metrics_<tag>.csv              all metrics in one table
  results/confusion_matrix_<tag>.png     both matrices as a figure
  results/metrics_<tag>_per_slice.csv    per-slice TP/FP/FN/TN
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Work whether this file sits in scripts/ or at the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "src")) \
    else os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src.dataset import BrainMRIDataset          # noqa: E402
from src.model import load_trained_model          # noqa: E402
from src.utils import get_device                  # noqa: E402

THRESHOLD = 0.5
EPS = 1e-7


def load_config():
    with open(os.path.join(_ROOT, "config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def safe_div(a, b):
    return float(a) / float(b) if b > 0 else 0.0


def metrics_from_counts(tp, fp, fn, tn):
    """Every metric the rubric asks for, from one confusion matrix."""
    # Empty slice, empty prediction: TP = FP = FN = 0, so precision,
    # recall, F1 and IoU are all 0/0 and mathematically undefined. The
    # segmentation convention is to score this as perfect agreement - the
    # model was asked whether a tumour was present, said no, and was right.
    # 481 of the 760 test slices are empty, so this choice moves the macro
    # average from about 0.30 to about 0.93.
    if tp + fp + fn == 0:
        return {
            "precision": 1.0,
            "recall_sensitivity": 1.0,
            "specificity": safe_div(tn, tn + fp),
            "f1_dice": 1.0,
            "iou_jaccard": 1.0,
            "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        }

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)          # also called sensitivity
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)  # identical to Dice
    iou = safe_div(tp, tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_dice": f1,
        "iou_jaccard": iou,
        "accuracy": accuracy,
    }


@torch.no_grad()
def collect(model, split_csv, image_size, device, batch_size):
    """Return a per-slice DataFrame of confusion-matrix counts."""
    ds = BrainMRIDataset(split_csv, "test", image_size)
    rows = pd.read_csv(split_csv)
    rows = rows[rows["split"] == "test"].reset_index(drop=True)
    if len(rows) != len(ds):
        raise SystemExit(
            f"Row mismatch: CSV has {len(rows)} test rows, dataset built "
            f"{len(ds)}.")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    recs = []
    idx = 0
    for images, masks in tqdm(loader, leave=False, desc="  evaluating"):
        images = images.to(device)
        preds = (torch.sigmoid(model(images)) > THRESHOLD).float().cpu().numpy()
        gts = (masks.numpy() > 0.5).astype(np.float32)

        for p, g in zip(preds, gts):
            p2, g2 = p[0], g[0]
            tp = float((p2 * g2).sum())
            fp = float((p2 * (1 - g2)).sum())
            fn = float(((1 - p2) * g2).sum())
            tn = float(((1 - p2) * (1 - g2)).sum())
            recs.append({
                "patient": rows.loc[idx, "patient"],
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "has_tumour": bool(g2.sum() > 0),
                "predicted_tumour": bool(p2.sum() > 0),
                "true_area": float(g2.sum()),
                "pred_area": float(p2.sum()),
                **metrics_from_counts(tp, fp, fn, tn),
            })
            idx += 1

    return pd.DataFrame(recs)


def plot_matrices(pix, slice_cm, out_path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ---- 1. pixel-level, row-normalised ----
    tp, fp, fn, tn = pix
    m = np.array([[tn, fp], [fn, tp]], dtype=float)
    mn = m / m.sum(axis=1, keepdims=True).clip(min=1)
    ax = axes[0]
    ax.imshow(mn, cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j]):,}\n({mn[i, j]*100:.2f}%)",
                    ha="center", va="center", fontsize=10.5,
                    color="white" if mn[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: no tumour", "Pred: tumour"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: no tumour", "True: tumour"])
    ax.set_title("Pixel-level confusion matrix\n(row-normalised)", fontsize=11)

    # ---- 2. slice-level detection ----
    s = np.array(slice_cm, dtype=float)   # [[tn, fp], [fn, tp]]
    sn = s / s.sum(axis=1, keepdims=True).clip(min=1)
    ax = axes[1]
    ax.imshow(sn, cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(s[i, j])}\n({sn[i, j]*100:.1f}%)",
                    ha="center", va="center", fontsize=10.5,
                    color="white" if sn[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted nothing", "Predicted tumour"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Slice has no tumour", "Slice has tumour"])
    ax.set_title("Slice-level detection\n(did it find anything at all?)", fontsize=11)

    fig.suptitle(f"Confusion matrices - model '{tag}', held-out test set",
                 fontsize=12.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="random")
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    root, ext = os.path.splitext(cfg["train"]["best_model_path"])
    model_path = os.path.join(_ROOT, f"{root}_{args.tag}{ext}")
    split_csv = args.split_csv or (
        os.path.join(_ROOT, "splits", "split_patient_random.csv")
        if args.tag == "random"
        else os.path.join(_ROOT, "splits", f"split_holdout_{args.tag}.csv"))

    for p in (model_path, split_csv):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}")

    device = get_device()
    print(f"Device : {device}")
    print(f"Model  : {model_path}")
    print(f"Split  : {split_csv}\n")

    model = load_trained_model(model_path, device=device)
    df = collect(model, split_csv, cfg["data"]["image_size"], device,
                 args.batch_size)

    tum = df[df["has_tumour"]]

    # ---------- MICRO (pooled pixels) ----------
    micro_all = metrics_from_counts(df.tp.sum(), df.fp.sum(),
                                    df.fn.sum(), df.tn.sum())
    micro_tum = metrics_from_counts(tum.tp.sum(), tum.fp.sum(),
                                    tum.fn.sum(), tum.tn.sum())

    # ---------- MACRO (per-slice average) ----------
    cols = ["precision", "recall_sensitivity", "specificity",
            "f1_dice", "iou_jaccard", "accuracy"]
    macro_all = {c: float(df[c].mean()) for c in cols}
    macro_tum = {c: float(tum[c].mean()) for c in cols}

    line = "=" * 78
    print(line)
    print(f"METRICS - model '{args.tag}'")
    print(line)
    print(f"Test slices : {len(df)}  ({len(tum)} contain tumour, "
          f"{len(df) - len(tum)} empty)")
    print(f"Patients    : {df['patient'].nunique()}\n")

    hdr = (f"{'metric':<22}{'micro all':>12}{'micro tumour':>14}"
           f"{'macro all':>12}{'macro tumour':>14}")
    print(hdr)
    print("-" * 78)
    pretty = {
        "precision": "Precision",
        "recall_sensitivity": "Recall / Sensitivity",
        "specificity": "Specificity",
        "f1_dice": "F1  ( = Dice )",
        "iou_jaccard": "IoU (Jaccard)",
        "accuracy": "Pixel accuracy",
    }
    for c in cols:
        print(f"{pretty[c]:<22}{micro_all[c]:>12.4f}{micro_tum[c]:>14.4f}"
              f"{macro_all[c]:>12.4f}{macro_tum[c]:>14.4f}")
    print("-" * 78)
    print("The headline number quoted throughout this project - 0.8176 - is")
    print("the MACRO TUMOUR F1/Dice column. That is the honest one.")

    # ---------- slice-level detection ----------
    s_tp = int(((df.has_tumour) & (df.predicted_tumour)).sum())
    s_fn = int(((df.has_tumour) & (~df.predicted_tumour)).sum())
    s_fp = int(((~df.has_tumour) & (df.predicted_tumour)).sum())
    s_tn = int(((~df.has_tumour) & (~df.predicted_tumour)).sum())
    det = metrics_from_counts(s_tp, s_fp, s_fn, s_tn)

    print("\n" + line)
    print("SLICE-LEVEL DETECTION  (did it find anything at all?)")
    print(line)
    print(f"{'':<26}{'Predicted nothing':>20}{'Predicted tumour':>20}")
    print(f"{'Slice HAS tumour':<26}{s_fn:>20}{s_tp:>20}")
    print(f"{'Slice has NO tumour':<26}{s_tn:>20}{s_fp:>20}")
    print("-" * 78)
    print(f"Detection precision   {det['precision']:.4f}")
    print(f"Detection recall      {det['recall_sensitivity']:.4f}   "
          f"<- {s_fn} tumour slices were missed entirely")
    print(f"Detection F1          {det['f1_dice']:.4f}")
    print(f"Detection specificity {det['specificity']:.4f}")
    if len(tum):
        print(f"\nMissed-slice rate: {s_fn}/{len(tum)} = "
              f"{100*s_fn/len(tum):.1f}% of tumour-bearing slices produce an "
              f"empty prediction.")

    # ---------- save ----------
    out_rows = []
    for name, d in [("micro_all_slices", micro_all),
                    ("micro_tumour_slices", micro_tum),
                    ("macro_all_slices", macro_all),
                    ("macro_tumour_slices", macro_tum),
                    ("slice_level_detection", det)]:
        out_rows.append({"aggregation": name, **{k: round(v, 4)
                                                 for k, v in d.items()}})
    res_dir = os.path.join(_ROOT, "results")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, f"metrics_{args.tag}.csv")
    pd.DataFrame(out_rows).to_csv(csv_path, index=False)

    per_slice = os.path.join(res_dir, f"metrics_{args.tag}_per_slice.csv")
    df.to_csv(per_slice, index=False)

    fig_path = os.path.join(res_dir, f"confusion_matrix_{args.tag}.png")
    plot_matrices((df.tp.sum(), df.fp.sum(), df.fn.sum(), df.tn.sum()),
                  [[s_tn, s_fp], [s_fn, s_tp]], fig_path, args.tag)

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {per_slice}")
    print(f"Saved -> {fig_path}")

    print("\n" + line)
    print("FOR YOUR VIVA - say this if asked why F1 was not reported earlier:")
    print(line)
    print("  For binary segmentation F1 and Dice are the same quantity:")
    print("    F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN) = Dice")
    print("  We reported Dice because it is the convention in segmentation.")
    print("  Precision and recall are still worth reporting because they")
    print("  decompose that score: they say whether our errors are")
    print("  over-segmentation or missed tumour, which Dice alone hides.")


if __name__ == "__main__":
    main()
