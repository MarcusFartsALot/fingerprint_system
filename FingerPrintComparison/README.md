# Fingerprint Enhancement System — BMDS2133 Image Processing (Mode A)

A Streamlit application built for the Fingerprint Enhancement System project
(Mode A: Comparative & Enhancement Study). It benchmarks the four classical
enhancement algorithms studied by the team **directly against fingerprint
images you upload** — there is no hardcoded/canned test image in the main
path — and exports a PDF report suitable for the assignment's Result &
Discussion / Appendix sections.

## Team algorithms

| Member | Algorithm | Module function |
|---|---|---|
| Ang Wei Ee | Gradient-Based Ridge Orientation Estimation | `algorithms.algo1_gradient_orientation` |
| Lam Yi Ming | Gabor Filtering | `algorithms.algo2_gabor` |
| Marcus Kong Mun Chun | Short-Time Fourier Transform (STFT) | `algorithms.algo3_stft` |
| Fong Jun Quan | Bilateral Edge-Preserving Denoising | `algorithms.algo4_bilateral_denoising` |

Algorithm 4 uses a `5 x 5` bilateral neighbourhood with colour sigma `12`
and spatial sigma `3`. It is the main denoising filter selected for the
Student Fingerprint Attendance application because it suppresses small
camera noise without averaging across strong fingerprint ridge boundaries.

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## App layout

### Tab 1 — Algorithm Comparison

**① Provide Fingerprint Sample(s)**
- **Upload real fingerprint image(s)** (the primary path) — choose one or
  more files, then pick an evaluation mode:
  - *Simulate degradation*: your upload is treated as the clean reference.
    A copy is synthetically degraded (blur/noise/smudges, all adjustable),
    and every algorithm is scored on how well it recovers the original.
  - *Direct*: your upload IS already the low-quality capture. Every
    algorithm runs on it as-is; since there's no separate clean reference,
    PSNR/SSIM are replaced by a no-reference **Ridge Clarity Score**.
- **Synthetic demo sample** — a secondary, clearly-labelled fallback for
  exploring the tool with no real image on hand. Do not report these
  numbers as your actual benchmark results.

**② Choose Algorithms** — pick which of the four to include.

**Run Benchmark** — for every sample:
1. A pseudo ground-truth ridge/valley mask is derived directly from the
   reference image using a binarisation pipeline independent of all four
   candidate algorithms (CLAHE + adaptive thresholding), so no algorithm is
   unfairly favoured (see `evaluation.py`).
2. Each selected algorithm enhances the (possibly degraded) image.
3. The enhanced output is Otsu-binarised and scored against the pseudo
   ground truth: **Confusion Matrix, Accuracy, Precision, Recall, F1**.
4. In *simulate* mode, PSNR/SSIM against the original upload are also
   computed.

**Results shown:**
- Metric summary table
- Sample gallery (reference / degraded-or-query / each algorithm's output),
  selectable per uploaded image
- Confusion matrices (ridge vs valley pixel classification)
- Grouped bar chart comparing Accuracy/Precision/Recall/F1
- PSNR/SSIM bars (simulate mode) or Ridge Clarity bars (direct mode)
- Intensity histograms
- Best-performing algorithm (highest average F1), highlighted

**Generate PDF Report / Download PDF Report** — exports everything above
into a formatted PDF (`pdf_report.py`, built with ReportLab).

### Tab 2 — Fingerprint Matching (Coming Soon)
Placeholder for the full graduation-verification application: import a
fingerprint → enhance → match against a student database → display Name,
Gender, Course, Grade Level, and Status (Graduated / Failed). Build this out
once the matching/database module is ready.

## Why a pseudo ground-truth mask instead of hand-labelled data?

Real fingerprint images — whether from a scanner, a phone photo, or a public
dataset — don't come with pixel-level ridge/valley labels, so a Confusion
Matrix or Accuracy/Precision/Recall/F1 can't be computed against them
directly. `evaluation.py` derives a reference-standard mask straight from
whichever image you upload, using a binarisation method that is *not* one of
the four candidates being compared (CLAHE + adaptive thresholding), so the
comparison stays fair. This methodology, its assumptions, and its
limitations should be described in the Methodology / Discussion sections of
your report — it is a documented workaround for the lack of ground truth,
not a claim of clinical-grade accuracy.

## File structure

```
app.py             Streamlit UI (two tabs)
algorithms.py       The four enhancement algorithms + shared helpers
evaluation.py        Pseudo ground-truth derivation + generic degradation
                      (the core, upload-driven benchmarking logic)
synthetic.py         OPTIONAL demo-only fallback (synthetic ridge pattern)
metrics_utils.py      Confusion matrix / Accuracy/Precision/Recall/F1 / PSNR/SSIM / charts
pdf_report.py         ReportLab-based PDF report builder
requirements.txt
```

## Extending Tab 2

When ready to build the real application, a reasonable next step is:
1. Pick one algorithm (or the best performer from Tab 1) as the production
   enhancement pipeline.
2. Add a minutiae or template-matching function (e.g. via `fingerprint`
   Python libraries, or a simple correlation/Hu-moments matcher for a class
   project).
3. Store student records (Name, Gender, Course, Grade level, Status) in a
   CSV/SQLite table keyed by an enrolled fingerprint template.
4. On upload, enhance → extract features → compare against every enrolled
   template → return the best match above a similarity threshold → display
   the matched record.
