"""
app.py
------
BMDS2133 Image Processing — Fingerprint Enhancement System (Mode A)

Tab 1: Algorithm Comparison — benchmarks the four enhancement algorithms
        studied by the team DIRECTLY against fingerprint image(s) the user
        uploads (no hardcoded/synthetic ground truth, and no synthetic
        degradation step, anywhere in the pipeline). Each algorithm runs
        standalone on every uploaded image exactly as uploaded, so the
        comparison reflects real enhancement performance, not recovery of
        an artificially-degraded copy.
        Supports batch uploads (e.g. 100 images at once): every image is
        scored individually against every algorithm, and the results are
        averaged per algorithm to determine the best performer.
        Produces Confusion Matrix, Accuracy/Precision/Recall/F1, a
        no-reference Ridge Clarity Score, histograms, and a PDF export.
Tab 2: Fingerprint Matching — placeholder ("Coming Soon") for the real
        graduation-verification application described in the brief.

Run with:  streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from algorithms import ALGORITHMS, binarize_otsu
from evaluation import derive_pseudo_ground_truth_mask, ridge_clarity_score, load_and_resize
from metrics_utils import (
    compute_classification_metrics,
    plot_confusion_matrices, plot_metric_bars, plot_clarity_bars,
    plot_histograms, plot_sample_gallery,
)
from pdf_report import build_pdf_report

IMAGE_LABEL = "Original (uploaded)"

st.set_page_config(page_title="Fingerprint Enhancement Studio", layout="wide")

st.title("Fingerprint Enhancement System")
st.caption("BMDS2133 Image Processing — Mode A: Comparative & Enhancement Study")

tab_compare, tab_app = st.tabs(["Algorithm Comparison", "Fingerprint Matching (Coming Soon)"])

# =============================================================================
# TAB 1 — ALGORITHM COMPARISON
# =============================================================================
with tab_compare:

    st.markdown(
        "Upload real fingerprint images and this tab will **process each one directly** through "
        "all four of the team's enhancement algorithms — standalone, with no synthetic "
        "degradation step — and generate the comparison metrics from them. For a full benchmark "
        "(e.g. a set of 100 fingerprint photos), select every file at once — each image is scored "
        "individually and the results below are the **average across every uploaded image**, "
        "per algorithm."
    )

    # ------------------------------------------------------------------ #
    # Step 1 — upload samples
    # ------------------------------------------------------------------ #
    st.subheader("Step 1: Upload Fingerprint Images")
    uploaded_files = st.file_uploader(
        "Upload one or more fingerprint images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} image(s) selected.")
        if len(uploaded_files) > 150:
            st.warning(
                "Large batches (150+ images) can take several minutes to process, since every "
                "image is run through every selected algorithm."
            )

    st.info(
        "Each algorithm enhances every uploaded image exactly as-is — no synthetic blur, noise, "
        "or degradation is applied, so the results reflect each algorithm's standalone "
        "enhancement performance on your real data. Confusion Matrix / Accuracy / Precision / "
        "Recall / F1 are computed against a pseudo ground-truth mask derived independently from "
        "each image, plus a no-reference Ridge Clarity Score."
    )

    st.subheader("Step 2: Choose Algorithms")
    selected_algos = st.multiselect(
        "Algorithms to include", options=list(ALGORITHMS.keys()), default=list(ALGORITHMS.keys()),
    )

    run = st.button("Run Benchmark", type="primary")

    # ------------------------------------------------------------------ #
    # Build sample list + run
    # ------------------------------------------------------------------ #
    if run:
        if not selected_algos:
            st.warning("Select at least one algorithm.")
        elif not uploaded_files:
            st.warning("Please upload at least one fingerprint image.")
        else:
            samples = []
            for f in uploaded_files:
                pil_img = Image.open(f).convert("L")
                arr = load_and_resize(np.array(pil_img), max_dim=300)
                gt_mask = derive_pseudo_ground_truth_mask(arr)
                samples.append({"image": arr, "gt_mask": gt_mask, "label": f.name})

            progress = st.progress(0, text="Running benchmark...")
            per_algo_records = {name: [] for name in selected_algos}
            total_steps = len(samples) * len(selected_algos)
            step = 0

            for sample in samples:
                sample["enhanced"] = {}
                sample["metrics"] = {}
                for name in selected_algos:
                    fn = ALGORITHMS[name]
                    t0 = time.perf_counter()
                    enhanced, aux = fn(sample["image"])
                    dt = time.perf_counter() - t0

                    pred_mask = binarize_otsu(enhanced)
                    cls = compute_classification_metrics(sample["gt_mask"], pred_mask)
                    rec = {**cls, "time": dt, "clarity": ridge_clarity_score(enhanced)}

                    per_algo_records[name].append(rec)
                    sample["enhanced"][name] = enhanced
                    sample["metrics"][name] = rec

                    step += 1
                    progress.progress(step / total_steps, text=f"{sample['label']} — {name.split(':')[0]}")

            progress.empty()

            # Aggregate (mean) across every uploaded image; keys are consistent within a run
            results = {}
            for name in selected_algos:
                recs = per_algo_records[name]
                keys = [k for k in recs[0].keys() if k != "confusion_matrix"]
                agg = {k: float(np.mean([r[k] for r in recs])) for k in keys}
                agg["confusion_matrix"] = sum(r["confusion_matrix"] for r in recs)
                results[name] = agg

            best_algo = max(results, key=lambda k: results[k]["f1"])

            st.session_state["benchmark_results"] = results
            st.session_state["benchmark_best_algo"] = best_algo
            st.session_state["benchmark_samples"] = samples
            label_preview = ", ".join(s["label"] for s in samples[:5])
            if len(samples) > 5:
                label_preview += f", … (+{len(samples) - 5} more)"
            st.session_state["benchmark_params"] = {
                "Sample source": "Uploaded real image(s)",
                "Evaluation mode": "Direct — each algorithm enhances the uploaded image as-is "
                                    "(no synthetic degradation)",
                "Images benchmarked": len(samples),
                "Sample labels": label_preview,
                "Algorithms tested": ", ".join(a.split(":")[0] for a in selected_algos),
            }
            st.success(
                f"Benchmark complete over {len(samples)} image(s). "
                f"Best algorithm by average F1-score: **{best_algo}**"
            )

    # ------------------------------------------------------------------ #
    # Render results (persists across reruns)
    # ------------------------------------------------------------------ #
    if "benchmark_results" in st.session_state:
        results = st.session_state["benchmark_results"]
        best_algo = st.session_state["benchmark_best_algo"]
        samples = st.session_state["benchmark_samples"]
        params = st.session_state["benchmark_params"]
        short_names = [n.split(":")[0] for n in results]

        st.divider()
        st.subheader("Metric Summary")
        st.markdown(
            f"**Best performing algorithm — highest average F1 across {len(samples)} image(s): "
            f"`{best_algo}`**"
        )

        table_rows_display = []
        for name, res in results.items():
            table_rows_display.append({
                "Algorithm": name,
                "Accuracy": f"{res['accuracy']:.3f}",
                "Precision": f"{res['precision']:.3f}",
                "Recall": f"{res['recall']:.3f}",
                "F1-score": f"{res['f1']:.3f}",
                "Clarity Score": f"{res['clarity']:.3f}",
                "Avg time (s)": f"{res['time']:.3f}",
            })
        st.dataframe(table_rows_display, use_container_width=True, hide_index=True)

        # Per-image breakdown — shows exactly how the average above was derived
        # across every uploaded image (useful appendix data for a report).
        with st.expander(f"Per-image metric breakdown ({len(samples)} images × {len(results)} algorithms)"):
            rows = []
            for s in samples:
                for name, rec in s["metrics"].items():
                    rows.append({
                        "Image": s["label"], "Algorithm": name.split(":")[0],
                        "Accuracy": round(rec["accuracy"], 3), "Precision": round(rec["precision"], 3),
                        "Recall": round(rec["recall"], 3), "F1": round(rec["f1"], 3),
                        "Clarity": round(rec["clarity"], 3), "Time (s)": round(rec["time"], 3),
                    })
            breakdown_df = pd.DataFrame(rows)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            csv_bytes = breakdown_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download per-image results (CSV)",
                data=csv_bytes,
                file_name="per_image_results.csv",
                mime="text/csv",
            )

        # ------------------------------------------------------------------ #
        # Confusion matrices — native dataframe (styled) per algorithm
        # ------------------------------------------------------------------ #
        st.subheader("Confusion Matrices")
        st.caption(f"Ridge (1) vs Valley (0) pixel classification, summed across all {len(samples)} image(s)")
        cm_cols = st.columns(len(results))
        for col, (name, res) in zip(cm_cols, results.items()):
            with col:
                st.caption(name.split(":")[0])
                cm_df = pd.DataFrame(
                    res["confusion_matrix"],
                    index=["Actual: Valley (0)", "Actual: Ridge (1)"],
                    columns=["Pred: Valley (0)", "Pred: Ridge (1)"],
                )
                st.dataframe(
                    cm_df.style.background_gradient(cmap="Blues").format("{:.0f}"),
                    use_container_width=True,
                )

        # ------------------------------------------------------------------ #
        # Classification metrics — native grouped bar chart
        # ------------------------------------------------------------------ #
        st.subheader("Classification Metrics")
        metric_cols = ["accuracy", "precision", "recall", "f1"]
        metric_df = pd.DataFrame(
            {m.capitalize(): [results[n][m] for n in results] for m in metric_cols},
            index=short_names,
        )
        st.bar_chart(metric_df, stack=False)

        # ------------------------------------------------------------------ #
        # Ridge clarity — native bar chart
        # ------------------------------------------------------------------ #
        st.subheader("Ridge Clarity")
        st.caption("No-reference ridge-orientation clarity score (0–1, higher = better)")
        clarity_df = pd.DataFrame({"Clarity": [results[n]["clarity"] for n in results]}, index=short_names)
        st.bar_chart(clarity_df)

        # Sample picker for gallery / histograms
        sample_labels = [s["label"] for s in samples]
        chosen_label = st.selectbox("View gallery / histogram for image:", sample_labels)
        chosen = next(s for s in samples if s["label"] == chosen_label)

        st.subheader(f"Sample Gallery — {chosen_label}")
        gallery_fig = plot_sample_gallery(chosen["image"], chosen["enhanced"], IMAGE_LABEL)
        st.pyplot(gallery_fig)

        st.subheader("Intensity Histograms")
        st.caption(f"Pixel intensity distribution for {chosen_label}")
        hist_sources = {IMAGE_LABEL: chosen["image"],
                         **{n.split(":")[0]: im for n, im in chosen["enhanced"].items()}}
        bins = np.linspace(0, 255, 33)
        bin_centers = ((bins[:-1] + bins[1:]) / 2).astype(int)
        hist_df = pd.DataFrame(
            {label: np.histogram(img, bins=bins)[0] for label, img in hist_sources.items()},
            index=bin_centers,
        )
        st.bar_chart(hist_df)

        st.divider()
        st.subheader("Export Report")
        if st.button("Generate PDF Report"):
            with st.spinner("Building PDF..."):
                header = ["Algorithm", "Accuracy", "Precision", "Recall", "F1", "Clarity", "Time (s)"]
                rows = [[n.split(":")[0], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
                         f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['clarity']:.3f}",
                         f"{r['time']:.3f}"] for n, r in results.items()]
                note = (
                    "Every algorithm was benchmarked standalone: each uploaded image was enhanced "
                    "as-is, with no synthetic degradation step. Output was scored against a pseudo "
                    "ground-truth ridge mask derived directly from the same image, plus a "
                    "no-reference ridge-orientation clarity score, then averaged across all "
                    "uploaded images."
                )
                # Rebuilt here as static matplotlib images (rather than the interactive Streamlit
                # components shown on screen) purely because the PDF export needs embeddable
                # figures — ReportLab can't embed a live Streamlit widget.
                cm_fig_pdf = plot_confusion_matrices(results)
                bar_fig_pdf = plot_metric_bars(results)
                clarity_fig_pdf = plot_clarity_bars(results)
                hist_images = {IMAGE_LABEL: chosen["image"], **chosen["enhanced"]}
                hist_fig_pdf = plot_histograms(hist_images)
                figures = {
                    "gallery": gallery_fig, "confusion": cm_fig_pdf, "metric_bars": bar_fig_pdf,
                    "quality_bars": clarity_fig_pdf, "histograms": hist_fig_pdf,
                }
                pdf_buf = build_pdf_report(header, rows, best_algo, figures, params, note)
                st.session_state["pdf_bytes"] = pdf_buf.getvalue()

        if "pdf_bytes" in st.session_state:
            st.download_button(
                "Download PDF Report",
                data=st.session_state["pdf_bytes"],
                file_name="fingerprint_algorithm_comparison_report.pdf",
                mime="application/pdf",
            )


# =============================================================================
# TAB 2 — REAL APPLICATION (COMING SOON)
# =============================================================================
with tab_app:
    st.subheader("Fingerprint Matching for Graduation Verification")
    st.info("Coming Soon — this module is under active development.")

    st.markdown(
        """
        This tab will host the full end-to-end application described in the assignment brief:

        1. **Import** a (possibly blurred / distorted) fingerprint image.
        2. **Enhance** it automatically using the best-performing algorithm identified in the
           *Algorithm Comparison* tab.
        3. **Match** the enhanced fingerprint against a student database.
        4. **Display** the matched student's record:
           - Name
           - Gender
           - Course
           - Grade level
           - Status (**Graduated** / **Failed**)

        ---
        **Planned architecture**
        - Enrolment step: store a reference (enhanced) fingerprint template + student record per
          student.
        - Matching step: minutiae or correlation-based matching between the query fingerprint and
          every enrolled template, returning the closest match above a similarity threshold.
        - Results dashboard: matched photo/template side-by-side with the student record and a
          confidence score.

        Check back once the team finalises the matching module and student database schema.
        """
    )
    st.button("Notify me when this is ready", disabled=True)