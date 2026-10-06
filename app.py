"""
app.py
======
TrustSeg - a brain tumour segmentation demo that reports how much it can
be trusted.

Run from the project root:

    set KMP_DUPLICATE_LIB_OK=TRUE
    venv\\Scripts\\python.exe -m streamlit run app.py

If streamlit is missing:

    venv\\Scripts\\python.exe -m pip install streamlit

WHAT THIS DEMONSTRATES
----------------------
A normal segmentation demo draws a red blob and stops. The problem with
that is a radiologist has no way to tell a confident outline from a guess -
both look equally certain on screen.

This app runs every slice through the model 8 times (the D4 symmetry group:
4 rotations, each with and without a mirror), maps every prediction back to
the original orientation, and measures how much the 8 views disagree. Views
that agree mean the decision is stable; views that disagree mean the model
is guessing.

That disagreement was validated offline on 279 tumour-bearing test slices:
referring the most-uncertain 20% to a human lifts Dice on the remainder
from 0.81 to 0.90. So the confidence number shown here is not decoration -
it predicts error.

UNCERTAINTY IS PRIMARY, AREA IS SECONDARY
-----------------------------------------
An earlier version of this file decided the verdict from the predicted area
first, and only consulted uncertainty if a tumour had been outlined. That
was backwards, and it produced the worst possible failure: on slices where
the averaged probability never crosses 0.5 but the individual views disagree
violently - the single most uncertain state the model can occupy - the app
reported a calm "NO TUMOUR DETECTED".

Measured on 224 tumour-bearing slices, 9.8% predict completely empty, and
half of those sit above the high-risk threshold. The verdict logic below
therefore branches on uncertainty FIRST, and an empty prediction with high
disagreement is reported as a possible missed finding, with the number of
views that did find something shown alongside it.

WHAT IT REPORTS
---------------
  * tumour mask overlay
  * per-pixel uncertainty heat map
  * tumour area (pixels, and mm^2 if you supply the pixel spacing)
  * hemisphere (left/right of the estimated brain midline)
  * a verdict with the reason, and how many of the 8 views found anything

HONESTY NOTES BUILT INTO THE UI
-------------------------------
  * Area is reported in pixels by default. mm^2 requires pixel spacing,
    which this dataset's TIFs do not carry, so it is an optional input.
  * Hemisphere is derived from the image's own brain midline, not from an
    anatomical atlas. Lobe-level localisation would need registration.
  * Dice is 1.0 when prediction and ground truth are both empty. That is a
    convention, not a result, and the UI says so.
  * The model segments FLAIR abnormality in lower-grade glioma. It does not
    classify tumour type, and it is not a diagnostic device.
"""

import os
import re
import sys

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="TrustSeg - Brain Tumour Segmentation",
                   page_icon=":brain:", layout="wide")

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.model import load_trained_model  # noqa: E402
from src.utils import make_overlay  # noqa: E402

THRESHOLD = 0.5
EPS = 1e-7
TTA_OPS = [(k, f) for f in (False, True) for k in (0, 1, 2, 3)]

MODEL_LABELS = {
    "random": "Baseline (trained on all 5 institutions)",
    "DU": "DU held out (never saw institution DU)",
    "HT": "HT held out (never saw institution HT)",
    "CS": "CS held out (never saw institution CS)",
    "FG": "FG held out (never saw institution FG)",
}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
@st.cache_data
def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_trained_model(model_path, device=device), device


@st.cache_data
def load_thresholds():
    """Derive the confidence bands from the offline validation run.

    Returns (low, high, source). The cut points are the 60th and 85th
    percentiles of view disagreement measured on the 279 tumour-bearing
    test slices, so the bands mean something empirical rather than being
    invented numbers.
    """
    path = os.path.join("results", "uncertainty_random_per_slice.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            t = df[df["has_tumour"]]
            if len(t) > 20:
                return (float(t["tta_disagree"].quantile(0.60)),
                        float(t["tta_disagree"].quantile(0.85)),
                        f"measured on {len(t)} validated test slices")
        except Exception:
            pass
    return 0.10, 0.25, "fallback defaults (run uncertainty.py to calibrate)"


@st.cache_data
def load_retention():
    path = os.path.join("results", "uncertainty_random_per_slice.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        t = df[df["has_tumour"]].sort_values("fg_entropy")
        n = len(t)
        rows = []
        for pct in (0, 10, 20, 30, 40, 50):
            keep = max(1, int(round(n * (1 - pct / 100))))
            rows.append({"% referred to radiologist": pct,
                         "slices kept": keep,
                         "Dice on kept slices":
                             round(float(t.iloc[:keep]["dice"].mean()), 4)})
        return pd.DataFrame(rows)
    except Exception:
        return None


@st.cache_data
def load_band_summary(low: float, high: float):
    """Mean Dice within each confidence band, computed from the same
    validated slices the bands came from. This is the evidence that the
    bands separate good predictions from bad ones."""
    path = os.path.join("results", "uncertainty_random_per_slice.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        t = df[df["has_tumour"]].copy()

        def band(u):
            if u < low:
                return "CONFIDENT"
            if u < high:
                return "REVIEW"
            return "HIGH RISK"

        t["band"] = t["tta_disagree"].map(band)
        g = (t.groupby("band")
             .agg(slices=("dice", "size"),
                  mean_dice=("dice", "mean"),
                  mean_instability=("tta_disagree", "mean"))
             .reindex(["CONFIDENT", "REVIEW", "HIGH RISK"]).reset_index())
        g["mean_dice"] = g["mean_dice"].round(3)
        g["mean_instability"] = g["mean_instability"].round(3)
        return g
    except Exception:
        return None


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def preprocess(rgb: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD,
                                                             np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def tta_fwd(x, k, flip):
    if flip:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k, dims=[2, 3])


def tta_inv(y, k, flip):
    y = torch.rot90(y, -k, dims=[2, 3])
    if flip:
        y = torch.flip(y, dims=[3])
    return y


@torch.no_grad()
def predict(model, device, x):
    """Return (mean probability map, 8 binary views)."""
    x = x.to(device)
    probs = []
    for k, flip in TTA_OPS:
        out = torch.sigmoid(model(tta_fwd(x, k, flip)))
        probs.append(tta_inv(out, k, flip))
    stack = torch.stack(probs, 0)[:, 0, 0].cpu().numpy()
    return stack.mean(0), (stack > THRESHOLD).astype(np.float32)


def dice_np(a, b):
    inter = float((a * b).sum())
    den = float(a.sum() + b.sum())
    return 1.0 if den == 0 else (2 * inter + EPS) / (den + EPS)


def disagreement(views):
    s = [dice_np(views[i], views[j])
         for i in range(len(views)) for j in range(i + 1, len(views))]
    return float(1 - np.mean(s)) if s else 0.0


def entropy_map(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def brain_midline(rgb: np.ndarray) -> float:
    """Estimate the x coordinate of the brain's centre line.

    The background in these scans is near-black, so thresholding away the
    dark border leaves the head, and its horizontal centre is a usable
    midline. This is image-derived, not anatomical.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = g > max(10, int(0.12 * g.max()))
    if mask.sum() < 50:
        return rgb.shape[1] / 2.0
    xs = np.where(mask.any(axis=0))[0]
    return float((xs.min() + xs.max()) / 2.0)


def heat_rgb(u: np.ndarray) -> np.ndarray:
    v = u / (u.max() + EPS)
    return cv2.applyColorMap((v * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)[:, :, ::-1]


def verdict_for(area_px: int, unc: float, n_views_found: int,
                low: float, high: float):
    """Decide the verdict. Uncertainty is checked FIRST; the predicted area
    only chooses the wording. An empty mask with high disagreement is a
    possible missed finding, never a clean negative."""
    if unc >= high:
        if area_px == 0:
            return ("HIGH RISK - POSSIBLE MISSED FINDING", "red",
                    f"No tumour was outlined, but the 8 views disagreed "
                    f"sharply and {n_views_found} of them did find an "
                    f"abnormality. A split vote is the least reliable state "
                    f"this model has. Do not read this as a negative result.")
        return ("HIGH RISK - REFER", "red",
                "The 8 views disagreed substantially. Slices in this band "
                "scored worst in validation (mean Dice 0.68). Treat this "
                "prediction as unreliable.")

    if unc >= low:
        if area_px == 0:
            return ("INCONCLUSIVE - NO OUTLINE, VIEWS DISAGREE", "orange",
                    f"No tumour was outlined, but {n_views_found} of the 8 "
                    f"views found something. This is inconclusive rather "
                    f"than negative.")
        return ("REVIEW RECOMMENDED", "orange",
                "The 8 views disagreed moderately, usually around the "
                "tumour boundary. The outline's extent is less reliable "
                "than its location.")

    if area_px == 0:
        return ("NO TUMOUR DETECTED", "blue",
                "None of the 8 views found a FLAIR abnormality. Unanimous "
                "agreement is not proof of absence - the model can be "
                "uniformly wrong, and on this dataset it is on about 5% of "
                "tumour-bearing slices, where there is no disagreement left "
                "to warn you. Review the full stack.")
    return ("CONFIDENT", "green",
            "All 8 augmented views agreed closely. Slices in this band had "
            "the highest Dice in validation (mean 0.93).")


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
cfg = load_config()
IMG = cfg["data"]["image_size"]
root, ext = os.path.splitext(cfg["train"]["best_model_path"])
LOW, HIGH, band_src = load_thresholds()

st.title("TrustSeg")
st.caption("Brain tumour segmentation that reports when it should not be "
           "trusted - FLAIR MRI, lower-grade glioma")

with st.sidebar:
    st.header("Model")
    available = [t for t in MODEL_LABELS if os.path.exists(f"{root}_{t}{ext}")]
    if not available:
        st.error(f"No trained models found in {os.path.dirname(root)}/")
        st.stop()
    tag = st.selectbox("Checkpoint", available,
                       format_func=lambda t: MODEL_LABELS.get(t, t))
    st.caption(f"`{root}_{tag}{ext}`")

    st.divider()
    st.header("Optional")
    spacing = st.number_input(
        "Pixel spacing (mm/pixel)", min_value=0.0, max_value=5.0,
        value=0.0, step=0.01,
        help="These TIFs carry no spacing metadata. Enter it from the "
             "original DICOM to convert area to mm². Leave 0 for pixels.")
    show_views = st.checkbox("Show all 8 TTA views", value=False)

    st.divider()
    st.caption(f"Confidence bands: {band_src}")
    st.caption(f"CONFIDENT < {LOW:.3f} · REVIEW < {HIGH:.3f} · "
               f"HIGH RISK ≥ {HIGH:.3f}")

model, device = get_model(f"{root}_{tag}{ext}")

tab_run, tab_evidence, tab_about = st.tabs(
    ["Analyse a scan", "Why trust the confidence score", "About"])

# ----------------------------------------------------------------------
with tab_run:
    c1, c2 = st.columns([1, 1])
    with c1:
        up = st.file_uploader("Upload an MRI slice",
                              type=["tif", "tiff", "png", "jpg", "jpeg"])
    with c2:
        gt_up = st.file_uploader("Ground-truth mask (optional, scores the "
                                 "prediction)",
                                 type=["tif", "tiff", "png", "jpg", "jpeg"])

    if up is None:
        st.info("Upload a slice from `kaggle_3m/` to begin. "
                "Example: `TCGA_CS_4944_20010208_10.tif`")
        st.stop()

    data = np.frombuffer(up.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read that image.")
        st.stop()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_disp = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR)

    with st.spinner("Running 8 test-time augmentation passes..."):
        mean_p, views = predict(model, device, preprocess(rgb, IMG))

    pred = (mean_p > THRESHOLD).astype(np.float32)
    unc = disagreement(views)
    ent = entropy_map(mean_p)
    area_px = int(pred.sum())
    n_found = int(sum(1 for v in views if v.sum() > 0))

    verdict, colour, reason = verdict_for(area_px, unc, n_found, LOW, HIGH)

    getattr(st, {"green": "success", "orange": "warning",
                 "red": "error", "blue": "info"}[colour])(
        f"**{verdict}**  ·  instability {unc:.3f}  ·  "
        f"{n_found} of 8 views found an abnormality\n\n{reason}")

    i1, i2, i3 = st.columns(3)
    with i1:
        st.image(rgb_disp, caption="Input slice", use_container_width=True)
    with i2:
        st.image(make_overlay(rgb_disp, pred, color=(255, 0, 0)),
                 caption=f"Predicted tumour ({area_px} px)",
                 use_container_width=True)
    with i3:
        st.image(heat_rgb(ent),
                 caption="Uncertainty (bright = model unsure)",
                 use_container_width=True)

    if area_px == 0 and n_found > 0:
        st.caption(f"The averaged probability never crossed {THRESHOLD}, so "
                   f"the mask is empty - but {n_found} of the 8 individual "
                   f"views did outline something. Tick **Show all 8 TTA "
                   f"views** in the sidebar to see them.")

    st.subheader("Measurements")
    m1, m2, m3, m4 = st.columns(4)

    if spacing > 0:
        area_txt = f"{area_px * spacing * spacing:.1f} mm²"
        area_help = f"{area_px} px × ({spacing} mm)²"
    else:
        area_txt = f"{area_px} px"
        area_help = "Enter pixel spacing in the sidebar for mm²"
    m1.metric("Tumour area", area_txt, help=area_help)

    brain_px = int((cv2.cvtColor(rgb_disp, cv2.COLOR_RGB2GRAY) > 10).sum())
    pct = 100 * area_px / brain_px if brain_px else 0
    m2.metric("Share of slice", f"{pct:.2f}%",
              help="Tumour pixels as a share of imaged head area")

    if area_px > 0:
        ys, xs = np.nonzero(pred)
        cx = xs.mean()
        mid = brain_midline(rgb_disp)
        side = "Left" if cx < mid else "Right"
        off = abs(cx - mid) / (rgb_disp.shape[1] / 2) * 100
        m3.metric("Hemisphere", f"{side} (image)", f"{off:.0f}% off midline",
                  help="Image-left/right from the brain's own midline. "
                       "Radiological convention may invert this; no atlas "
                       "registration is applied.")
    else:
        m3.metric("Hemisphere", "n/a")

    m4.metric("View instability", f"{unc:.3f}",
              help="1 − mean pairwise Dice across the 8 TTA views. "
                   "0 = perfect agreement. This is the live confidence "
                   "signal; it holds its sign whether or not a tumour is "
                   "present.")

    m = re.search(r"_(\d+)\.(tif|tiff|png|jpe?g)$", up.name, re.I)
    if m:
        st.caption(f"Slice index parsed from filename: **{m.group(1)}** "
                   f"· patient `{up.name.rsplit('_', 1)[0]}`")

    if gt_up is not None:
        gdata = np.frombuffer(gt_up.read(), np.uint8)
        gm = cv2.imdecode(gdata, cv2.IMREAD_GRAYSCALE)
        if gm is not None:
            gm = cv2.resize(gm, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
            g = (gm > 127).astype(np.float32)
            d = dice_np(pred, g)
            both_empty = (g.sum() == 0 and pred.sum() == 0)

            st.subheader("Scored against the provided mask")
            s1, s2, s3 = st.columns(3)
            s1.metric("Dice", f"{d:.4f}")
            inter = float((pred * g).sum())
            union = float(pred.sum() + g.sum() - inter)
            s2.metric("IoU", f"{(inter + EPS) / (union + EPS):.4f}")
            s3.metric("True area", f"{int(g.sum())} px")

            if both_empty:
                st.caption("Both masks are empty, so Dice is 1.0 by "
                           "convention - the overlap of two empty sets is "
                           "defined as perfect. This is not a measure of "
                           "performance on this slice.")
            st.image(make_overlay(make_overlay(rgb_disp, g, color=(0, 255, 0)),
                                  pred, alpha=0.35, color=(255, 0, 0)),
                     caption="Green = ground truth · Red = prediction",
                     width=380)

    if show_views:
        st.subheader("The 8 test-time augmentation views")
        st.caption("Each is the same slice rotated/mirrored, predicted, then "
                   "mapped back. Disagreement between these is the "
                   "confidence signal.")
        cols = st.columns(4)
        for i, v in enumerate(views):
            found = "found" if v.sum() > 0 else "empty"
            cols[i % 4].image(make_overlay(rgb_disp, v, color=(255, 0, 0)),
                              caption=f"view {i + 1} - {found}",
                              use_container_width=True)

# ----------------------------------------------------------------------
with tab_evidence:
    st.subheader("The confidence score predicts error - here is the proof")
    st.write(
        "Offline, every one of the 279 tumour-bearing test slices was ranked "
        "by uncertainty. Referring the most uncertain slices to a human and "
        "letting the model keep the rest produces this curve. A rising curve "
        "means the model is discarding exactly the cases it was about to get "
        "wrong.")
    tb = load_retention()
    if tb is not None:
        c1, c2 = st.columns([1, 1])
        c1.dataframe(tb, hide_index=True, use_container_width=True)
        c2.line_chart(tb.set_index("% referred to radiologist")
                      ["Dice on kept slices"])
        st.success("Referring the most uncertain 20% - ranked by foreground "
                   "entropy - lifts Dice on the remainder from 0.81 to 0.90.")
    else:
        st.info("Run `uncertainty.py` to generate this evidence.")

    st.divider()
    st.subheader("Do the verdict bands actually separate good from bad?")
    bs = load_band_summary(LOW, HIGH)
    if bs is not None:
        st.dataframe(bs, hide_index=True, use_container_width=True)
        st.caption("Computed on the same validated slices the thresholds "
                   "came from. Dice falls and instability rises monotonically "
                   "across the bands, which is what makes the live verdict "
                   "meaningful rather than cosmetic.")
    else:
        st.info("Run `uncertainty.py` to generate this table.")

    st.divider()
    st.subheader("Two scores, two jobs")
    st.markdown(
        "- **View instability (shown live in the app).** Holds its sign on "
        "every slice, tumour or not - which is what deployment needs, since "
        "you cannot know in advance whether a tumour is present. "
        "ρ = −0.63 on tumour slices, −0.88 overall.\n"
        "- **Foreground entropy (used in the retention curve above).** "
        "Stronger where a tumour is present (ρ = −0.86), but it divides by "
        "predicted area, so it inverts on empty slices by construction. "
        "Valid for the offline analysis, unsuitable as a live indicator.\n\n"
        "The 0.81 → 0.90 figure belongs to foreground entropy. The number on "
        "the Analyse tab is view instability. They are not interchangeable.")

    st.divider()
    st.subheader("Known limits")
    st.markdown(
        "- Trained on 110 patients. The learning curve projects a ceiling "
        "near 0.92 Dice and does not saturate until roughly 96 training "
        "patients, so this model is still data-limited.\n"
        "- **The blind spot.** About 10% of tumour-bearing slices are "
        "predicted completely empty. The app red-flags half of them, because "
        "the views disagree. The other half - roughly 5% of all "
        "tumour-bearing slices - are missed with *zero* disagreement: all 8 "
        "views are unanimously wrong, so there is no signal left to warn "
        "anyone. Disagreement-based uncertainty cannot detect a failure the "
        "model is uniformly confident about, and this is the method's "
        "fundamental limit.\n"
        "- **The cost of flagging early.** On 491 tumour-free slices, 5.5% "
        "are shown as high risk. This is a deliberate trade: on this dataset "
        "a false alarm costs a second look, a missed tumour costs more. The "
        "false alarms cluster by patient rather than scattering evenly.\n"
        "- Per-institution test sets of 14-45 patients are too small to "
        "detect scanner domain shift: observed spread 0.131 against an "
        "explainable effect of 0.032.\n"
        "- Single modality (FLAIR), 2D slices, lower-grade glioma only.")

# ----------------------------------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        "1. The slice is resized to 256×256 and normalised with ImageNet "
        "statistics.\n"
        "2. It is passed through a U-Net with a ResNet-34 encoder **8 "
        "times** - four 90° rotations, each with and without a mirror.\n"
        "3. Each prediction is mapped back to the original orientation, so "
        "all 8 should coincide.\n"
        "4. Their average is the mask; their disagreement is the confidence "
        "score.\n\n"
        "The model was trained with flips and 90° rotations, so it has every "
        "reason to be invariant to them. Disagreement therefore reflects "
        "genuine model uncertainty rather than an unfamiliar input.")

    st.divider()
    st.subheader("Why uncertainty decides the verdict, not the mask")
    st.markdown(
        "The mask is produced by averaging the 8 views and thresholding at "
        "0.5. That average can land below the threshold even when several "
        "individual views clearly outline something - a split vote. Deciding "
        "the verdict from the mask alone would present exactly that case, "
        "the most uncertain state the model has, as a clean negative. So the "
        "verdict branches on disagreement first, and the mask only chooses "
        "the wording.")

    st.divider()
    st.subheader("Not a diagnostic device")
    st.markdown(
        "Research prototype built on the public TCGA lower-grade glioma "
        "cohort (Buda, Saha & Mazurowski, *Computers in Biology and "
        "Medicine*, 2019). It segments FLAIR abnormality. It does not "
        "classify tumour type, grade, or malignancy, and it must not be "
        "used for clinical decisions.")
