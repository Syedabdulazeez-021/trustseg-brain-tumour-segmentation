"""
analyze_learning_curve.py
=========================
Turns results/learning_curve.csv into the project's central quantitative
result, and tests whether the earlier cross-site findings survive it.

Run from the project root (seconds, no GPU work):

    venv\\Scripts\\python.exe scripts/analyze_learning_curve.py

WHAT IT ESTABLISHES
-------------------
1. THE CURVE. How test Dice grows with the number of training patients,
   with the test set held fixed so every point is comparable. A saturating
   curve is fitted so the shape can be stated numerically rather than
   eyeballed, and the point of diminishing returns is read off it.

2. THE NOISE FLOOR. At each training size the same experiment was repeated
   with 3 seeds against the SAME test set. The spread between those repeats
   is pure training-side noise - a lower bound on how small an effect this
   dataset can resolve at all.

3. THE VERDICT ON THE CROSS-SITE STUDY. Every leave-one-hospital-out and
   matched-control model is placed on the curve by its own training-set
   size. The residual (observed minus predicted) says how much of each
   model's score the curve already accounts for. If the residuals are
   scattered around zero and no larger than the noise floor, then neither
   site identity nor training volume is driving those numbers - test-set
   composition is, and the honest conclusion is that per-site test sets of
   14-45 patients are too small to detect domain shift.

OUTPUTS
-------
  results/learning_curve.png          the curve with error bars and fit
  results/curve_vs_crosssite.png      LOHO/control models placed on it
  results/curve_analysis.csv          residual table
"""

import os
import sys

import numpy as np
import pandas as pd

# This file lives in scripts/, one level below the project root. Put the
# root on sys.path (for `src`) and this folder (for sibling scripts such
# as `train`), so imports work exactly as they did when it sat at root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LC_CSV = os.path.join("results", "learning_curve.csv")
SUMMARY_CSV = os.path.join("results", "domain_shift_summary.csv")
RESULTS = "results"

# Training-set sizes of the nine cross-site models.
# train patients = (patients in that split marked "train")
CROSS_SITE = [
    # name,          arm,        n_train, observed tumour Dice
    ("LOHO - DU",    "loho",      56, 0.7052),
    ("LOHO - HT",    "loho",      65, 0.7420),
    ("LOHO - CS",    "loho",      80, 0.7173),
    ("LOHO - FG",    "loho",      82, 0.7979),
    ("Control - DU", "control",   55, 0.6915),
    ("Control - HT", "control",   65, 0.7268),
    ("Control - CS", "control",   80, 0.8228),
    ("Control - FG", "control",   82, 0.7291),
]
BASELINE = ("Baseline", "baseline", 72, 0.8176)


def saturating(n, a, b, c):
    """Dice = a - b / (n + c). Rises and flattens, which is the shape
    learning curves actually take."""
    return a - b / (n + c)


def fit_curve(n, d):
    """Least-squares fit of the saturating form, using a coarse grid for c
    so no optimiser dependency is needed beyond numpy."""
    best = None
    for c in np.linspace(1, 120, 400):
        x = 1.0 / (n + c)
        A = np.vstack([np.ones_like(x), -x]).T
        coef, res, *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ coef
        sse = float(((d - pred) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, coef[0], coef[1], c)
    _, a, b, c = best
    return float(a), float(b), float(c)


def main():
    if not os.path.exists(LC_CSV):
        raise SystemExit(f"Cannot find {LC_CSV}. Run learning_curve.py first.")

    lc = pd.read_csv(LC_CSV)
    agg = (lc.groupby("n_train_patients")["test_dice_tumour"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "dice", "std": "sd", "count": "runs"}))

    n = agg["n_train_patients"].values.astype(float)
    d = agg["dice"].values
    a, b, c = fit_curve(n, d)

    noise = float(agg["sd"].mean())

    line = "=" * 74
    print(line)
    print("1. THE LEARNING CURVE  (fixed test set, 3 seeds per point)")
    print(line)
    print(f"{'Train patients':>15}{'Dice':>10}{'SD':>9}{'Fitted':>10}{'Gain':>9}")
    print("-" * 74)
    prev = None
    for i, r in agg.iterrows():
        nn = r["n_train_patients"]
        fit = saturating(nn, a, b, c)
        gain = "" if prev is None else f"{r['dice'] - prev:+.4f}"
        print(f"{int(nn):>15}{r['dice']:>10.4f}{r['sd']:>9.4f}"
              f"{fit:>10.4f}{gain:>9}")
        prev = r["dice"]
    print("-" * 74)
    print(f"Fitted form : Dice = {a:.4f} - {b:.2f}/(n + {c:.1f})")
    print(f"Ceiling as n -> infinity : {a:.4f}")

    # Point of diminishing returns: where the curve gains < 0.01 per 15 patients.
    knee = None
    for nn in range(10, 400):
        if saturating(nn + 15, a, b, c) - saturating(nn, a, b, c) < 0.01:
            knee = nn
            break
    if knee:
        print(f"Diminishing returns from  : ~{knee} training patients "
              f"(<0.01 Dice per extra 15)")

    print(f"\nNOISE FLOOR (same data size, same test set, different seed)")
    print(f"  mean SD across the 3 repeats = {noise:.4f}")
    print(f"  => effects smaller than about {2 * noise:.3f} Dice cannot be")
    print(f"     resolved by this dataset at all.")

    # ---------------- 2. Place the cross-site models on the curve ----------
    rows = []
    for name, arm, n_tr, obs in [BASELINE] + CROSS_SITE:
        pred = saturating(n_tr, a, b, c)
        rows.append({
            "model": name, "arm": arm, "n_train_patients": n_tr,
            "observed_dice": obs, "curve_predicted": pred,
            "residual": obs - pred,
        })
    res = pd.DataFrame(rows)

    print("\n" + line)
    print("2. DO THE CROSS-SITE RESULTS SURVIVE THE CURVE?")
    print(line)
    print("Each model is placed on the curve by its OWN training-set size.")
    print("Residual = what the curve does NOT explain.\n")
    print(f"{'model':<15}{'arm':<10}{'n_train':>9}{'observed':>11}"
          f"{'curve':>9}{'residual':>11}")
    print("-" * 74)
    for _, r in res.iterrows():
        print(f"{r['model']:<15}{r['arm']:<10}{int(r['n_train_patients']):>9}"
              f"{r['observed_dice']:>11.4f}{r['curve_predicted']:>9.4f}"
              f"{r['residual']:>+11.4f}")
    print("-" * 74)

    cs = res[res["arm"] != "baseline"]
    loho = cs[cs["arm"] == "loho"]["residual"]
    ctrl = cs[cs["arm"] == "control"]["residual"]
    spread = float(cs["observed_dice"].max() - cs["observed_dice"].min())
    curve_span = float(saturating(cs["n_train_patients"].max(), a, b, c)
                       - saturating(cs["n_train_patients"].min(), a, b, c))

    print(f"Residual SD (all 8 cross-site models) : {cs['residual'].std():.4f}")
    print(f"Mean residual, LOHO arm               : {loho.mean():+.4f}")
    print(f"Mean residual, control arm            : {ctrl.mean():+.4f}")
    print(f"Difference (LOHO - control)           : "
          f"{loho.mean() - ctrl.mean():+.4f}")

    print("\n" + line)
    print("3. THE DECISIVE NUMBERS")
    print(line)
    print(f"Observed spread across the 8 cross-site models : {spread:.4f}")
    print(f"Spread the curve predicts over that size range : {curve_span:.4f}")
    print(f"Training-side noise floor (2 SD)               : {2 * noise:.4f}")
    print()
    if spread > max(curve_span, 2 * noise) * 1.5:
        print("READING: the spread between cross-site models is far larger")
        print("than either the training-size effect or the seed noise can")
        print("explain. The remaining variable is WHICH PATIENTS ended up in")
        print("each test set. With 14-45 test patients per site, test-set")
        print("sampling noise dominates, and a site effect of the size usually")
        print("reported in the literature could not be detected here even if")
        print("it existed.")
    else:
        print("READING: the spread is comparable to the curve and noise, so")
        print("training size plausibly accounts for the cross-site results.")

    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(os.path.join(RESULTS, "curve_analysis.csv"), index=False)

    # ---------------- Figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(n.min() * 0.8, max(n.max(), 85) * 1.05, 200)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(n, d, yerr=agg["sd"].values, fmt="o", capsize=4,
                color="#1B2B33", label="measured (3 seeds)", zorder=3)
    ax.plot(grid, saturating(grid, a, b, c), "-", color="#E8871E",
            label=f"fit: {a:.3f} - {b:.1f}/(n+{c:.0f})", zorder=2)
    if knee:
        ax.axvline(knee, ls="--", color="#5B6B73", lw=1,
                   label=f"diminishing returns ~{knee}")
    ax.set_xlabel("Number of training patients")
    ax.set_ylabel("Test Dice (tumour-bearing slices)")
    ax.set_title("How much data does LGG segmentation actually need?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(RESULTS, "learning_curve.png")
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, saturating(grid, a, b, c), "-", color="#E8871E",
            lw=2, label="learning curve (fixed test set)", zorder=1)
    ax.fill_between(grid,
                    saturating(grid, a, b, c) - 2 * noise,
                    saturating(grid, a, b, c) + 2 * noise,
                    color="#E8871E", alpha=0.15,
                    label="±2 SD training noise", zorder=0)
    for arm, colour, marker in [("loho", "#133C55", "o"),
                                ("control", "#A8443A", "s")]:
        sub = res[res["arm"] == arm]
        ax.scatter(sub["n_train_patients"], sub["observed_dice"],
                   c=colour, marker=marker, s=70, zorder=3,
                   label=f"{arm} models")
    bl = res[res["arm"] == "baseline"]
    ax.scatter(bl["n_train_patients"], bl["observed_dice"], c="#1C7293",
               marker="*", s=230, zorder=4, label="baseline")
    ax.set_xlabel("Number of training patients")
    ax.set_ylabel("Test Dice (tumour-bearing slices)")
    ax.set_title("Cross-site models placed on the learning curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(RESULTS, "curve_vs_crosssite.png")
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    print(f"\nSaved -> {p1}")
    print(f"Saved -> {p2}")
    print(f"Saved -> {os.path.join(RESULTS, 'curve_analysis.csv')}")


if __name__ == "__main__":
    main()
