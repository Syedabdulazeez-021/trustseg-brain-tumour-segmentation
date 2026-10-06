# TrustSeg — Brain Tumour Segmentation That Reports When It Should Not Be Trusted

U-Net with a ResNet-34 encoder segmenting FLAIR abnormality in lower-grade glioma,
with a test-time-augmentation uncertainty score that **predicts its own errors**.

A conventional segmentation demo draws a red blob and stops: a clinician has no way
to tell a confident outline from a guess, because both look equally certain on
screen. This project runs every slice through the model eight times under the D4
symmetry group (four 90° rotations, each with and without a mirror), maps each
prediction back to the original orientation, and measures how much the eight views
disagree. That disagreement turns out to rank errors well enough to be useful:
referring the most-uncertain 20% of slices to a human lifts Dice on the remainder
from **0.81 to 0.90**. The same pipeline was used to test whether a per-hospital
performance gap is real domain shift or just a small-sample artefact — and found
that **it is an artefact**, with the apparent site effect (−0.0020) an order of
magnitude below the measurement noise floor (0.057).

> ### ⚠️ What is *not* in this repository
> - **Model checkpoints (2.33 GB, 25 files)** — regenerate with `scripts/train.py`
>   and `scripts/learning_curve.py`. See *Reproducing the results* below.
> - **The dataset (1 GB, 7,860 files)** — download separately, see *Dataset*.
> - **`splits/lc/` (15 CSVs)** — regenerate identically; the seeds are fixed.
>
> Everything else — all source, all config, all result CSVs and figures, and the
> nine top-level split definitions — is here. The repository is meant to be
> readable and reproducible without the large files.

---

## Headline results

| Measurement | Value | Source |
|---|---:|---|
| Baseline Dice (tumour-bearing test slices, mixed-hospital split) | **0.8176** | `results/domain_shift_summary.csv` |
| Baseline Dice (all test slices, incl. tumour-free) | 0.9278 | `results/domain_shift_summary.csv` |
| Mean site effect, leave-one-hospital-out vs. matched control | **−0.0020** | `results/domain_shift_paired.csv` |
| Measurement noise floor (2 × mean seed-to-seed SD) | **0.057** | `results/learning_curve.csv` |
| Dice after referring the most-uncertain 20% of slices | **0.81 → 0.90** | `results/uncertainty_random_per_slice.csv` |
| Training patients | 110 total (80 train / 15 val / 15 test) | `config/config.yaml` |

**The domain-shift result is the one worth reading twice.** Holding out a whole
hospital drops Dice by up to 0.112, which looks like textbook scanner domain shift.
But holding out a hospital also removes its patients from training, so the drop
confounds *site* with *training-set size*. Pairing each leave-one-hospital-out run
against a size-matched control split that draws from all hospitals isolates the two
effects: the mean site effect is **−0.0020**. Retraining at a fixed data size with
only the seed changed moves Dice by 0.0283 on average, so effects smaller than about
**0.057** (2 SD) cannot be resolved by this dataset at all — the measured site effect
is roughly 28× below that floor. The apparent domain shift is dominated by having
trained on fewer patients, not by the scanner. No per-site *p* value reaches
significance (0.21–0.78). With this cohort size, the honest conclusion is that the
experiment **cannot detect** site effects, not that none exist.

### Figures

| File | Shows |
|---|---|
| `results/learning_curve.png` | Dice vs. training-set size, 5 sizes × 3 seeds |
| `results/curve_vs_crosssite.png` | Cross-site drops against the learning curve — the core argument |
| `results/uncertainty_random_retention.png` | Dice vs. % referred to a radiologist |
| `results/uncertainty_random_examples.png` | Qualitative overlays with uncertainty maps |
| `results/curves_*.png` | Training curves per experiment arm (9 arms) |

---

## Dataset — not included

**LGG Segmentation Dataset** — 110 lower-grade glioma patients from The Cancer
Genome Atlas (TCGA), via The Cancer Imaging Archive (TCIA). Pre-operative FLAIR MRI
with manually traced abnormality masks.

- **Download:** <https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>
- **Size:** ~1 GB, 7,860 `.tif` files across 110 patient folders

The data is **deliberately excluded** from this repository — it is large, and it is
TCGA/TCIA data carrying its own terms of use, which this MIT licence does not cover.

After downloading, unzip so the project root contains a `kaggle_3m/` folder with the
110 `TCGA_*` patient sub-folders:

```
brain_tumor_segmentation/
└── kaggle_3m/
    ├── TCGA_CS_4941_19960909/
    │   ├── TCGA_CS_4941_19960909_1.tif
    │   ├── TCGA_CS_4941_19960909_1_mask.tif
    │   └── ...
    └── ... (109 more)
```

If you unzip elsewhere, point `data.raw_root` in `config/config.yaml` at it.

---

## Setup

Requires Python 3.11+ and, for practical training times, an NVIDIA GPU.
Developed on Windows with Python 3.13 and CUDA 12.6.

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a GPU build of PyTorch (the plain `pip install torch` gives you CPU-only, which
works but is slow):

```bat
venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 ^
    --index-url https://download.pytorch.org/whl/cu126
```

### ⚠️ `KMP_DUPLICATE_LIB_OK` — set this in every shell that runs torch

PyTorch and OpenCV each ship their own copy of the Intel OpenMP runtime. On Windows,
loading both aborts the process with *"OMP: Error #15: Initializing libiomp5md.dll,
but found libiomp5md.dll already initialized"*. Set this first, in every shell:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
```

PowerShell: `$env:KMP_DUPLICATE_LIB_OK = "TRUE"` · bash: `export KMP_DUPLICATE_LIB_OK=TRUE`

### Verify the install

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
venv\Scripts\python.exe scripts\smoke_test.py
```

Thirteen checks covering dependencies, config, checkpoints, dataset, model loading,
TTA invertibility, and a full eight-pass inference. It tells you exactly what is
wrong and how to fix it, rather than letting you discover it mid-demo.

---

## Running the demo

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
venv\Scripts\python.exe -m streamlit run app.py
```

Opens at <http://localhost:8501>. Upload any slice from `kaggle_3m/` — for example
`TCGA_CS_4941_19960909_12.tif` — and optionally its `_mask.tif` to score the
prediction. The app shows the mask overlay, a per-pixel uncertainty heat map, tumour
area, hemisphere relative to the image's own brain midline, and a verdict of
CONFIDENT / REVIEW / HIGH RISK / NO TUMOUR with the reasoning behind it.

Requires at least one checkpoint in `results/`. Train one first (below).

---

## Reproducing the results

Run everything **from the project root**, with `KMP_DUPLICATE_LIB_OK=TRUE` set.
Steps 3–6 are long: roughly **25 GPU-hours total** on a single consumer GPU, the
bulk of it in step 5.

### 1 — Build the patient-level split

```bat
venv\Scripts\python.exe data\prepare_data.py
```
Writes `data/split.csv` — 110 patients into 80 train / 15 val / 15 test, seed 42.

### 2 — Build the nine experiment splits

```bat
venv\Scripts\python.exe scripts\make_splits.py
venv\Scripts\python.exe scripts\make_control_splits.py
```
`make_splits.py` writes the mixed-hospital baseline (`split_patient_random.csv`) and
four leave-one-hospital-out splits (`split_holdout_{DU,HT,CS,FG}.csv`).
`make_control_splits.py` writes four size-matched controls
(`split_control_{DU,HT,CS,FG}.csv`) that draw from all hospitals — these are what
separate a site effect from a sample-size effect.

### 3 — Train the nine models (~9 GPU-hours)

```bat
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_patient_random.csv --tag random
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_holdout_DU.csv --tag DU
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_holdout_HT.csv --tag HT
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_holdout_CS.csv --tag CS
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_holdout_FG.csv --tag FG
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_control_DU.csv --tag ctrlDU
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_control_HT.csv --tag ctrlHT
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_control_CS.csv --tag ctrlCS
venv\Scripts\python.exe scripts\train.py --split_csv splits\split_control_FG.csv --tag ctrlFG
```
Each writes `results/best_model_<tag>.pth` and `results/curves_<tag>.png`.

### 4 — Evaluate domain shift

```bat
venv\Scripts\python.exe scripts\evaluate_domain_shift.py
```
Writes `results/domain_shift_summary.csv` (per-arm Dice),
`results/domain_shift_per_patient.csv`, and `results/domain_shift_paired.csv`
(the leave-one-hospital-out vs. control pairing, with *p* values).

### 5 — Learning curve (~15 GPU-hours, 15 models)

```bat
venv\Scripts\python.exe scripts\learning_curve.py
```
Trains at 15/30/45/60/75 training patients × 3 seeds. Writes
`results/learning_curve.csv`, `results/learning_curve.png`, the checkpoints in
`results/lc/`, and the subset splits in `splits/lc/`.

### 6 — Tie the two together

```bat
venv\Scripts\python.exe scripts\analyze_learning_curve.py
```
Fits the curve, predicts what each cross-site arm *should* score given its training
size alone, and reports the residual — the part attributable to the site. Writes
`results/curve_analysis.csv` and `results/curve_vs_crosssite.png`.

### 7 — Uncertainty analysis

```bat
venv\Scripts\python.exe scripts\uncertainty.py --tag random
```
Eight-pass TTA over the test set. Writes
`results/uncertainty_random_per_slice.csv` (the file the demo reads to calibrate its
confidence bands), `results/uncertainty_random_retention.png`, and
`results/uncertainty_random_examples.png`. Add `--tag DU` etc. for other arms.

### Optional — single-model evaluation and sample extraction

```bat
venv\Scripts\python.exe evaluate.py --model_path results\best_model_random.pth
venv\Scripts\python.exe scripts\find_tumor_slices.py --n 5
```

---

## Repository layout

```
├── app.py                      Streamlit demo (TrustSeg)
├── evaluate.py                 Single-model test-set evaluation
├── config/config.yaml          All hyperparameters and paths
├── src/                        Library code
│   ├── dataset.py              Dataset, augmentation, ImageNet normalisation
│   ├── model.py                U-Net + ResNet-34 (segmentation_models_pytorch)
│   ├── losses.py               Dice + BCE
│   ├── metrics.py              Dice, IoU, sensitivity, specificity
│   └── utils.py                Overlays, curves, device handling
├── scripts/                    Experiment entry points (run from project root)
│   ├── train.py                Train one model
│   ├── make_splits.py          Baseline + leave-one-hospital-out splits
│   ├── make_control_splits.py  Size-matched control splits
│   ├── learning_curve.py       15 models: 5 sizes × 3 seeds
│   ├── evaluate_domain_shift.py  Per-arm Dice + paired site/volume effects
│   ├── analyze_learning_curve.py Curve fit vs. cross-site residuals
│   ├── uncertainty.py          8-pass TTA uncertainty + retention curve
│   ├── find_tumor_slices.py    Extract demo slices from the dataset
│   └── smoke_test.py           13-check pre-flight for the demo
├── data/
│   ├── prepare_data.py         Builds the patient-level split
│   └── split.csv               110-patient train/val/test assignment
├── splits/                     Nine experiment split definitions (CSV)
├── results/                    Result CSVs and figures (checkpoints gitignored)
├── tests/test_pipeline.py      Unit tests
├── notebooks/colab_train.ipynb Colab training notebook
└── docs/                       Project report (added separately)
```

---

## Known limitations

Stated plainly, because a model that reports its own confidence should not overstate
its own competence.

**1. A ~5% silent miss rate that the uncertainty score cannot catch.**
On 224 tumour-bearing slices, 9.8% produce a completely empty mask. Just over half
are correctly flagged — the eight views split, instability runs 0.14–0.75, and the
demo reports HIGH RISK — POSSIBLE MISSED FINDING. But the remaining **~5% are missed
with all eight views in unanimous agreement** (instability 0.000), including one
slice with a 2,897-pixel tumour. TTA measures *disagreement*; it is blind by
construction to a model that is uniformly wrong. Unanimity is not evidence of
absence.

**2. ~5.5% false-alarm rate on tumour-free slices.**
Checking uncertainty before predicted area is what fixes limitation 1, and it costs
false positives: of 491 genuinely tumour-free slices, 27 (5.5%) are shown as HIGH
RISK and 4 more as REVIEW. 93.7% are still correctly reported as clean. The alarms
cluster by patient — up to a third of one patient's healthy slices. This is a
deliberate trade: a false alarm is cheaper than a missed tumour.

**3. 110 patients is too few to settle the domain-shift question.**
Seed-to-seed retraining alone moves Dice by 0.0283 (SD), giving a 0.057 noise floor
against a measured site effect of 0.0020, on per-hospital test sets of just 14–45
patients. The learning curve has not saturated at 75 training
patients and projects a ceiling near 0.92 Dice at roughly 96. The right conclusion
is *"underpowered"*, not *"no effect"*.

**4. Scope.** Single modality (FLAIR), 2D slices evaluated independently, lower-grade
glioma only. The model segments FLAIR abnormality; it does not classify tumour type,
grade, or malignancy. Hemisphere is derived from the image's own brain midline, not
an anatomical atlas. Area is reported in pixels because these TIFs carry no spacing
metadata.

**5. Not a diagnostic device.** Research prototype. Not validated for clinical use
and must not inform clinical decisions.

---

## Citation

If you use this dataset, cite the paper it accompanies:

```bibtex
@article{buda2019association,
  title   = {Association of genomic subtypes of lower-grade gliomas with shape
             features automatically extracted by a deep learning algorithm},
  author  = {Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal = {Computers in Biology and Medicine},
  volume  = {109},
  pages   = {218--225},
  year    = {2019},
  doi     = {10.1016/j.compbiomed.2019.05.002}
}
```

Data originates from The Cancer Genome Atlas (TCGA) and is distributed through
The Cancer Imaging Archive (TCIA).

---

## Author

**Abdul Azeez**

B.Tech Computer Science & Engineering (Artificial Intelligence)
Amrita Vishwa Vidyapeetham, Amritapuri

## Licence

MIT — see [LICENSE](LICENSE). Covers the source code only, not the dataset.
