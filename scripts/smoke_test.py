"""
smoke_test.py
=============
Checks everything app.py depends on WITHOUT needing a browser or Streamlit.

Run from the project root:

    set KMP_DUPLICATE_LIB_OK=TRUE
    venv\\Scripts\\python.exe scripts/smoke_test.py

If every check passes, the demo will work. If one fails, it says exactly
what is wrong and how to fix it - which is faster than discovering it in
front of a panel.
"""

import os
import sys
import traceback

# This file lives in scripts/, one level below the project root. Put the
# root on sys.path (for `src`) and this folder (for sibling scripts such
# as `train`), so imports work exactly as they did when it sat at root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS, FAIL, WARN = [], [], []


def check(name, fn, fatal=False):
    try:
        msg = fn()
        PASS.append(f"{name}" + (f"  -> {msg}" if msg else ""))
        return True
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        (FAIL if fatal else WARN).append(f"{name}\n      {type(e).__name__}: {e}")
        if fatal:
            print(f"\nFATAL in '{name}':\n{tb}")
        return False


def main():
    print("=" * 70)
    print("TrustSeg demo - pre-flight check")
    print("=" * 70)

    # ---- 1. dependencies ----
    def dep_streamlit():
        import streamlit
        v = streamlit.__version__
        major, minor = (int(x) for x in v.split(".")[:2])
        if (major, minor) < (1, 40):
            raise RuntimeError(
                f"streamlit {v} is old; app.py uses use_container_width "
                f"(needs >= 1.40). Fix: pip install -U streamlit")
        return f"streamlit {v}"

    check("streamlit installed and recent enough", dep_streamlit, fatal=True)
    check("torch + CUDA", lambda: (
        __import__("torch").__version__ +
        (" (CUDA available)" if __import__("torch").cuda.is_available()
         else " (CPU only - demo will be slow but works)")), fatal=True)
    check("opencv", lambda: __import__("cv2").__version__, fatal=True)
    check("pandas / numpy / yaml", lambda: (
        __import__("pandas").__version__), fatal=True)

    # ---- 2. project files ----
    import yaml
    cfg = None

    def load_cfg():
        nonlocal cfg
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        return f"image_size={cfg['data']['image_size']}"

    if not check("config/config.yaml readable", load_cfg, fatal=True):
        return report()

    root, ext = os.path.splitext(cfg["train"]["best_model_path"])

    def find_models():
        tags = ["random", "DU", "HT", "CS", "FG"]
        found = [t for t in tags if os.path.exists(f"{root}_{t}{ext}")]
        if not found:
            raise FileNotFoundError(
                f"No checkpoints matching {root}_<tag>{ext}")
        return f"{len(found)} found: {', '.join(found)}"

    check("trained model checkpoints", find_models, fatal=True)

    def calib():
        p = "results/uncertainty_random_per_slice.csv"
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing - the app will fall back to uncalibrated "
                f"confidence bands. Fix: run uncertainty.py")
        import pandas as pd
        d = pd.read_csv(p)
        t = d[d["has_tumour"]]
        lo = t["tta_disagree"].quantile(0.60)
        hi = t["tta_disagree"].quantile(0.85)
        return (f"{len(t)} tumour slices; bands CONFIDENT<{lo:.3f} "
                f"REVIEW<{hi:.3f}")

    check("confidence bands calibrated from real data", calib)

    def sample_slice():
        import glob
        pats = sorted(glob.glob("kaggle_3m/TCGA_*"))
        if not pats:
            raise FileNotFoundError("kaggle_3m/ not found next to app.py")
        tifs = [f for f in glob.glob(os.path.join(pats[0], "*.tif"))
                if "_mask" not in f]
        if not tifs:
            raise FileNotFoundError(f"No .tif slices in {pats[0]}")
        return f"{len(pats)} patients; e.g. {os.path.basename(tifs[0])}"

    check("dataset present for the live demo", sample_slice)

    # ---- 3. the actual pipeline ----
    import numpy as np
    import torch
    import cv2
    from src.model import load_trained_model
    from src.dataset import IMAGENET_MEAN, IMAGENET_STD
    from src.utils import make_overlay

    IMG = cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {}

    def load_model():
        mp = f"{root}_random{ext}"
        if not os.path.exists(mp):
            import glob
            cands = glob.glob(f"{root}_*{ext}")
            if not cands:
                raise FileNotFoundError("no checkpoint to load")
            mp = cands[0]
        state["model"] = load_trained_model(mp, device=device)
        n = sum(p.numel() for p in state["model"].parameters())
        return f"{os.path.basename(mp)}, {n:,} params, on {device}"

    if not check("model loads onto device", load_model, fatal=True):
        return report()

    def tta_roundtrip():
        """The critical correctness check: rotating a prediction back must
        land it exactly where it started. If this is wrong, every
        uncertainty number in the project is wrong."""
        x = torch.randn(1, 1, 16, 16)
        for flip in (False, True):
            for k in (0, 1, 2, 3):
                y = x
                if flip:
                    y = torch.flip(y, dims=[3])
                y = torch.rot90(y, k, dims=[2, 3])
                y = torch.rot90(y, -k, dims=[2, 3])
                if flip:
                    y = torch.flip(y, dims=[3])
                if not torch.allclose(x, y):
                    raise AssertionError(f"k={k} flip={flip} did not invert")
        return "all 8 transforms invert exactly"

    check("TTA transforms are exactly invertible", tta_roundtrip, fatal=True)

    def end_to_end():
        import glob
        tifs = [f for f in glob.glob("kaggle_3m/TCGA_*/*.tif")
                if "_mask" not in f]
        if tifs:
            bgr = cv2.imread(tifs[0], cv2.IMREAD_COLOR)
            src = os.path.basename(tifs[0])
        else:
            bgr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            src = "synthetic image (dataset not found)"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img = cv2.resize(rgb, (IMG, IMG))
        a = img.astype(np.float32) / 255.0
        a = (a - np.array(IMAGENET_MEAN, np.float32)) / \
            np.array(IMAGENET_STD, np.float32)
        x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device)

        outs = []
        with torch.no_grad():
            for flip in (False, True):
                for k in (0, 1, 2, 3):
                    xt = torch.flip(x, dims=[3]) if flip else x
                    xt = torch.rot90(xt, k, dims=[2, 3])
                    p = torch.sigmoid(state["model"](xt))
                    p = torch.rot90(p, -k, dims=[2, 3])
                    if flip:
                        p = torch.flip(p, dims=[3])
                    outs.append(p)
        stack = torch.stack(outs, 0)[:, 0, 0].cpu().numpy()
        mean_p = stack.mean(0)
        views = (stack > 0.5).astype(np.float32)

        eps = 1e-7
        sims = []
        for i in range(8):
            for j in range(i + 1, 8):
                inter = float((views[i] * views[j]).sum())
                den = float(views[i].sum() + views[j].sum())
                sims.append(1.0 if den == 0 else (2 * inter + eps) / (den + eps))
        unc = 1 - float(np.mean(sims))

        pred = (mean_p > 0.5).astype(np.float32)
        ov = make_overlay(cv2.resize(rgb, (IMG, IMG)), pred, color=(255, 0, 0))
        if ov.shape != (IMG, IMG, 3):
            raise AssertionError(f"overlay shape wrong: {ov.shape}")

        heat = cv2.applyColorMap(
            ((mean_p / (mean_p.max() + eps)) * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO)[:, :, ::-1]
        if heat.shape != (IMG, IMG, 3):
            raise AssertionError("heatmap shape wrong")

        return (f"{src} -> {int(pred.sum())} tumour px, "
                f"instability {unc:.3f}")

    check("full 8-pass pipeline + overlay + heatmap", end_to_end, fatal=True)

    def midline():
        import glob
        tifs = [f for f in glob.glob("kaggle_3m/TCGA_*/*.tif")
                if "_mask" not in f]
        if not tifs:
            return "skipped (no dataset)"
        rgb = cv2.cvtColor(cv2.imread(tifs[0], cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = g > max(10, int(0.12 * g.max()))
        if mask.sum() < 50:
            raise AssertionError("brain mask empty - midline would default")
        xs = np.where(mask.any(axis=0))[0]
        mid = (xs.min() + xs.max()) / 2
        off = abs(mid - rgb.shape[1] / 2) / rgb.shape[1] * 100
        if off > 25:
            raise AssertionError(
                f"estimated midline is {off:.0f}% off image centre - "
                f"hemisphere labels may be unreliable")
        return f"midline at x={mid:.0f} ({off:.1f}% off centre)"

    check("brain midline estimation (hemisphere label)", midline)

    def page_icon():
        """st.set_page_config rejects a bare word as page_icon on some
        versions."""
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        if 'page_icon="brain"' in src:
            raise RuntimeError(
                'page_icon="brain" may error on some Streamlit versions. '
                'Fix: change it to page_icon=":brain:" in app.py')
        return "ok"

    check("app.py page_icon is safe", page_icon)

    report()


def report():
    print("\n" + "=" * 70)
    for p in PASS:
        print(f"  PASS  {p}")
    for w in WARN:
        print(f"  WARN  {w}")
    for f in FAIL:
        print(f"  FAIL  {f}")
    print("=" * 70)
    if FAIL:
        print(f"\n{len(FAIL)} blocking problem(s). Fix these before demoing.")
        sys.exit(1)
    elif WARN:
        print(f"\nNo blockers. {len(WARN)} warning(s) - the app will run, "
              f"but read them.")
    else:
        print("\nEverything checks out. Launch with:")
        print("  venv\\Scripts\\python.exe -m streamlit run app.py")


if __name__ == "__main__":
    main()
