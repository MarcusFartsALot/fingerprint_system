"""
Fingerprint Enhancement System — Main Application (Module D: User Interface)
================================================================================
Author (team): Member D

Integrates:
    Module A — Image Acquisition & Gradient-Based Ridge Orientation Estimation
    Module B — Feature Detection (Minutiae + Singular Points)
    Module C — Pattern Classification & Matching/Identification
    Module D — This file: Streamlit UI, local SQLite database, PDF reporting,
               batch processing, and video-frame ingestion.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from modules import database
from modules.acquisition_orientation import (generate_vector_overlay,
                                              run_acquisition_pipeline)
from modules.calibration import calibrate_image
from modules.classification import classify_pattern
from modules.matching import identify
from modules.minutiae_detection import (detect_minutiae, draw_minutiae_overlay,
                                         yolo_model_available)
from modules.report import generate_report
from modules.singular_points import detect_singular_points

# ----------------------------------------------------------------
# PAGE CONFIGURATION & THEMING
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Fingerprint Enhancement System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: 700; color: #00ff88; margin-bottom: 6px; }
    .subtitle { color: #9a9a9a; margin-bottom: 20px; }
    .section-header { font-size: 18px; font-weight: 600; border-left: 4px solid #00ff88; padding-left: 10px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

database.init_db()

st.markdown('<div class="main-title">🩻 Fingerprint Enhancement System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BMDS2133 Image Processing — Mode B Innovative Solution '
            '(Topic 3: Fingerprint Enhancement System)</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------
# SIDEBAR — SHARED HYPERPARAMETERS (Module A controls, reused everywhere)
# ----------------------------------------------------------------
st.sidebar.header("🎛️ Algorithmic Hyperparameters")

st.sidebar.subheader("1. Contrast Enhancement (CLAHE)")
clip_limit = st.sidebar.slider("Clip Limit", 1.0, 10.0, 3.0, 0.5)
tile_grid = st.sidebar.slider("Tile Grid Size", 4, 32, 8, 4)

st.sidebar.subheader("2. Orientation Estimation")
block_size = st.sidebar.slider("Window Block Size (W)", 8, 32, 16, 2)
gaussian_sigma = st.sidebar.slider("Gaussian Smoothing Sigma", 0.5, 5.0, 3.0, 0.5)

st.sidebar.subheader("3. Adaptive Binarisation")
adaptive_block = st.sidebar.slider("Adaptive Threshold Block Size", 5, 31, 15, 2)
adaptive_c = st.sidebar.slider("Constant Subtraction (C)", -5, 10, 3, 1)

st.sidebar.subheader("4. Feature Detection")
use_yolo = st.sidebar.checkbox("Prefer trained YOLO minutiae model", value=True,
                                help="Falls back automatically to the classical "
                                     "crossing-number detector if no trained "
                                     "weights are found at models/minutiae_yolo.pt")
st.sidebar.caption(f"YOLO weights found: {'✅ yes' if yolo_model_available() else '❌ no (using classical detector)'}")

st.sidebar.markdown("---")
st.sidebar.caption("Local database: SQLite (`data/fingerprints.db`)")
st.sidebar.caption(f"Records stored: {len(database.get_all_records(hydrate=False))}")


def run_full_pipeline(raw_matrix: np.ndarray) -> dict:
    """Runs Modules A -> B -> C end to end and returns every artefact the
    UI/report/database need."""
    pipeline = run_acquisition_pipeline(
        raw_matrix, clip_limit, tile_grid, adaptive_block, adaptive_c,
        block_size, gaussian_sigma,
    )

    calib = calibrate_image(pipeline["raw"], pipeline["theta_field"], block_size=24)

    singular_points = detect_singular_points(pipeline["theta_field"], block_size=block_size)

    bgr_for_yolo = cv2.cvtColor(pipeline["raw"], cv2.COLOR_GRAY2BGR)
    minutiae = detect_minutiae(pipeline["binary"], pipeline["theta_field"],
                                image_bgr=bgr_for_yolo, prefer_yolo=use_yolo)

    classification = classify_pattern(singular_points)

    minutiae_overlay = draw_minutiae_overlay(pipeline["binary"], minutiae)
    singular_overlay = pipeline["vector_overlay"].copy()
    for p in singular_points:
        colour = (0, 165, 255) if p.kind == "core" else (255, 0, 255)
        cv2.drawMarker(singular_overlay, (p.x, p.y), colour,
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)

    return {
        **pipeline,
        "calibration": calib,
        "singular_points": singular_points,
        "minutiae": minutiae,
        "classification": classification,
        "minutiae_overlay": minutiae_overlay,
        "singular_overlay": singular_overlay,
    }


def render_pipeline_results(result: dict, key_prefix: str = ""):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-header">1. Enhanced Ridge Structure</div>', unsafe_allow_html=True)
        st.image(result["binary"], use_container_width=True, caption="CLAHE + Adaptive Binarisation")
    with col2:
        st.markdown('<div class="section-header">2. Minutiae Detection</div>', unsafe_allow_html=True)
        st.image(result["minutiae_overlay"], use_container_width=True, channels="BGR",
                 caption=f"{len(result['minutiae'])} minutiae "
                         f"({'YOLO' if result['minutiae'] and result['minutiae'][0].source == 'yolo' else 'classical CN'})")
    with col3:
        st.markdown('<div class="section-header">3. Singular Points</div>', unsafe_allow_html=True)
        st.image(result["singular_overlay"], use_container_width=True, channels="BGR",
                 caption="Orange = core, Magenta = delta")

    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pattern Type", result["classification"].pattern_type)
    m2.metric("Minutiae Count", len(result["minutiae"]))
    m3.metric("Estimated DPI", f"{result['calibration'].estimated_dpi:.0f}")
    m4.metric("Cores / Deltas", f"{result['classification'].num_cores} / {result['classification'].num_deltas}")
    st.info(result["classification"].explanation)


# ----------------------------------------------------------------
# TABS
# ----------------------------------------------------------------
tab_single, tab_batch, tab_video, tab_db, tab_dashboard = st.tabs([
    "🔍 Single Analysis", "📁 Batch Processing", "🎞️ Video Ingestion",
    "🗄️ Database & Identification", "📈 Analytics Dashboard",
])

# ================================================================
# TAB 1 — SINGLE IMAGE ANALYSIS
# ================================================================
with tab_single:
    uploaded_file = st.file_uploader("📤 Upload a fingerprint image", type=["png", "jpg", "jpeg", "tif", "bmp"],
                                      key="single_upload")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_matrix = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        with st.spinner("Running acquisition → calibration → detection → classification..."):
            result = run_full_pipeline(raw_matrix)

        st.session_state["last_result"] = result
        st.session_state["last_raw"] = raw_matrix
        st.session_state["last_filename"] = uploaded_file.name

        render_pipeline_results(result)

        st.markdown("---")
        st.subheader("💾 Save to Local Database")
        c1, c2, c3 = st.columns([2, 2, 1])
        subject_name = c1.text_input("Subject name", key="save_name")
        subject_id = c2.text_input("Subject ID (optional)", key="save_id")
        if c3.button("Save Record", use_container_width=True):
            if not subject_name:
                st.warning("Please enter a subject name before saving.")
            else:
                os.makedirs(database.IMAGE_DIR, exist_ok=True)
                image_path = os.path.join(database.IMAGE_DIR,
                                           f"{subject_name}_{datetime.now():%Y%m%d%H%M%S}.png")
                cv2.imwrite(image_path, raw_matrix)
                database.insert_record(
                    subject_name, subject_id, image_path,
                    result["classification"].pattern_type, result["minutiae"],
                    result["singular_points"], result["calibration"].estimated_dpi,
                )
                st.success(f"Saved record for '{subject_name}'.")

        st.markdown("---")
        st.subheader("📄 Export PDF Report")
        if st.button("Generate Report"):
            with st.spinner("Building PDF report..."):
                out_path = os.path.join(tempfile.gettempdir(), f"fingerprint_report_{datetime.now():%Y%m%d%H%M%S}.pdf")
                generate_report(
                    out_path, subject_name or "Unnamed", subject_id or "",
                    result["raw"], result["enhanced"], result["minutiae_overlay"],
                    result["singular_overlay"], result["classification"],
                    result["calibration"].estimated_dpi, len(result["minutiae"]),
                )
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download Report PDF", f, file_name="fingerprint_report.pdf",
                                    mime="application/pdf")
    else:
        st.info("Upload a fingerprint image to begin the 3-stage analysis pipeline "
                "(enhancement → feature detection → classification).")

# ================================================================
# TAB 2 — BATCH PROCESSING (Extra Effort: bulk ingestion)
# ================================================================
with tab_batch:
    st.write("Upload multiple fingerprint images at once for bulk processing "
             "(Extra Effort: bulk/folder-style ingestion).")
    batch_files = st.file_uploader("📤 Upload multiple images", type=["png", "jpg", "jpeg", "tif", "bmp"],
                                    accept_multiple_files=True, key="batch_upload")

    if batch_files:
        if st.button(f"Process {len(batch_files)} images"):
            progress = st.progress(0.0)
            summary_rows = []
            for i, f in enumerate(batch_files):
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    continue
                res = run_full_pipeline(raw)
                summary_rows.append({
                    "File": f.name,
                    "Pattern": res["classification"].pattern_type,
                    "Minutiae": len(res["minutiae"]),
                    "Cores": res["classification"].num_cores,
                    "Deltas": res["classification"].num_deltas,
                    "Est. DPI": round(res["calibration"].estimated_dpi, 1),
                })
                progress.progress((i + 1) / len(batch_files))

            st.session_state["batch_summary"] = summary_rows
            st.success(f"Processed {len(summary_rows)} images.")

    if "batch_summary" in st.session_state:
        st.dataframe(st.session_state["batch_summary"], use_container_width=True)

# ================================================================
# TAB 3 — VIDEO INGESTION (Extra Effort: video stream processing)
# ================================================================
with tab_video:
    st.write("Upload a short video (e.g. a fingerprint card being scanned frame-by-frame) "
             "and run the pipeline on any extracted frame (Extra Effort: video-stream ingestion).")
    video_file = st.file_uploader("📤 Upload video", type=["mp4", "avi", "mov"], key="video_upload")

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.caption(f"Video contains {frame_count} frames.")

        frame_idx = st.slider("Select frame to analyse", 0, max(frame_count - 1, 0), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()

        if ok:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            st.image(gray_frame, caption=f"Frame {frame_idx}", use_container_width=True, clamp=True)

            if st.button("Analyse this frame"):
                with st.spinner("Running pipeline on selected frame..."):
                    result = run_full_pipeline(gray_frame)
                render_pipeline_results(result, key_prefix="video")
        else:
            st.error("Could not read the selected frame.")

        os.unlink(tmp_path)

# ================================================================
# TAB 4 — DATABASE & IDENTIFICATION
# ================================================================
with tab_db:
    st.subheader("🗄️ Stored Records")
    records = database.get_all_records()

    if records:
        table_view = [{
            "ID": r["id"], "Subject": r["subject_name"], "Pattern": r["pattern_type"],
            "Minutiae": r["num_minutiae"], "Cores": r["num_cores"], "Deltas": r["num_deltas"],
            "Est. DPI": round(r["estimated_dpi"], 1) if r["estimated_dpi"] else None,
            "Saved": r["created_at"],
        } for r in records]
        st.dataframe(table_view, use_container_width=True)

        del_col1, del_col2 = st.columns([3, 1])
        del_id = del_col1.number_input("Record ID to delete", min_value=0, step=1)
        if del_col2.button("Delete Record"):
            database.delete_record(int(del_id))
            st.rerun()
    else:
        st.info("No records saved yet. Save records from the Single Analysis tab.")

    st.markdown("---")
    st.subheader("🔎 Identify a Fingerprint Against the Database")
    query_file = st.file_uploader("📤 Upload a query fingerprint", type=["png", "jpg", "jpeg", "tif", "bmp"],
                                   key="query_upload")

    if query_file is not None and records:
        file_bytes = np.asarray(bytearray(query_file.read()), dtype=np.uint8)
        query_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if st.button("Run Identification"):
            with st.spinner("Extracting features and comparing against all stored records..."):
                query_result = run_full_pipeline(query_raw)
                match_results = identify(query_result["minutiae"], query_result["singular_points"], records)

            st.session_state["last_query_result"] = query_result
            st.session_state["last_match_results"] = match_results

            render_pipeline_results(query_result, key_prefix="query")

            st.markdown("### Ranked Matches")
            match_table = [{
                "Rank": i + 1, "Subject": m.subject_name, "Score": round(m.score, 3),
                "Matched Pairs": m.matched_pairs, "Decision": "✅ MATCH" if m.is_match else "no match",
            } for i, m in enumerate(match_results[:10])]
            st.dataframe(match_table, use_container_width=True)

            if match_results and match_results[0].is_match:
                st.success(f"Best match: **{match_results[0].subject_name}** "
                           f"(score {match_results[0].score:.3f})")
            else:
                st.warning("No confident match found in the database (all scores below threshold).")
    elif query_file is not None and not records:
        st.warning("No records in the database yet — save some fingerprints first.")

# ================================================================
# TAB 5 — ANALYTICS DASHBOARD
# ================================================================
with tab_dashboard:
    st.subheader("📈 Database Analytics")
    records = database.get_all_records(hydrate=False)

    if not records:
        st.info("No data yet — process and save fingerprints to populate the dashboard.")
    else:
        total = len(records)
        avg_minutiae = np.mean([r["num_minutiae"] for r in records])
        avg_dpi = np.mean([r["estimated_dpi"] for r in records if r["estimated_dpi"]])

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", total)
        m2.metric("Avg. Minutiae / Print", f"{avg_minutiae:.1f}")
        m3.metric("Avg. Estimated DPI", f"{avg_dpi:.0f}")

        st.markdown("#### Pattern Type Distribution")
        distribution = database.pattern_type_distribution()
        st.bar_chart(distribution)

        st.markdown("#### All Records")
        st.dataframe(records, use_container_width=True)
