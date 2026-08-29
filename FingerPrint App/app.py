"""Ridge-preserving student attendance fingerprint system.

Run from this directory with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import importlib
import sqlite3
import uuid
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import fingerprint_processing as fingerprint_module
import matching as matching_module
import database as database_module
import ui as ui_module

# Streamlit reruns this file in the same Python process. Reload the lightweight
# persistence module so newly added administration functions are immediately
# available instead of leaving an older module cached in sys.modules.
database_module = importlib.reload(database_module)
ui_module = importlib.reload(ui_module)

from database import (
    DATA_DIR,
    DATABASE_PATH,
    REFERENCE_DIR,
    TEMPLATE_DIR,
    add_fingerprint_templates,
    attendance_records,
    create_session,
    dashboard_statistics,
    delete_session,
    delete_student,
    enrol_student,
    get_student,
    initialise_database,
    list_sessions,
    list_students,
    list_templates,
    local_now,
    mark_attendance,
    recent_audit_events,
    session_roster,
    update_session,
    update_student,
)
# Streamlit can retain an older imported module while hot-reloading this file.
# Reload only when the processing module's public interface has changed.
_REQUIRED_PIPELINE_SCHEMA_VERSION = "phone-ridge-preserving-v10"
_PROCESSING_EXPORTS = (
    "PIPELINE_SCHEMA_VERSION",
    "PipelineResult",
    "generate_demo_fingerprint",
    "process_fingerprint",
    "clahe_contrast_enhancement",
    "bilateral_ridge_denoising",
    "mild_unsharp_ridge_enhancement",
)
_processing_module_reloaded = False
if (
    any(not hasattr(fingerprint_module, name) for name in _PROCESSING_EXPORTS)
    or getattr(fingerprint_module, "PIPELINE_SCHEMA_VERSION", None)
    != _REQUIRED_PIPELINE_SCHEMA_VERSION
):
    fingerprint_module = importlib.reload(fingerprint_module)
    _processing_module_reloaded = True

PipelineResult = fingerprint_module.PipelineResult
generate_demo_fingerprint = fingerprint_module.generate_demo_fingerprint
process_fingerprint = fingerprint_module.process_fingerprint
if (
    _processing_module_reloaded
    or getattr(matching_module, "MATCHING_PIPELINE_SCHEMA_VERSION", None)
    != fingerprint_module.PIPELINE_SCHEMA_VERSION
):
    matching_module = importlib.reload(matching_module)
from matching import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MATCH_THRESHOLD,
    IdentificationResult,
    attendance_status,
    capture_quality_issue,
    identify_student,
    save_enhanced_template,
    save_reference_capture,
)
from reporting import build_attendance_pdf
from ui import apply_theme, page_header, pipeline_strip, result_banner, sidebar_brand, stat_card


st.set_page_config(
    page_title="Student Fingerprint Attendance",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)
_PIPELINE_STATE_VERSION_KEY = "_pipeline_schema_version"
if (
    _processing_module_reloaded
    or st.session_state.get(_PIPELINE_STATE_VERSION_KEY)
    != fingerprint_module.PIPELINE_SCHEMA_VERSION
):
    for state_key in ("last_scan", "studio_result", "enrolment_preview"):
        st.session_state.pop(state_key, None)
    st.session_state[_PIPELINE_STATE_VERSION_KEY] = fingerprint_module.PIPELINE_SCHEMA_VERSION
apply_theme()
initialise_database()


IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
DEMO_SCAN_DIR = DATA_DIR / "demo_scans"
PROGRAMMES = [
    "Bachelor of Software Engineering (Honours)",
    "Bachelor of Computer Science (Honours)",
    "Bachelor of Information Technology (Honours)",
    "Bachelor of Data Science (Honours)",
]


def fmt_datetime(value: str | None, short: bool = False) -> str:
    if value is None or str(value).strip().lower() in {"", "nan", "nat", "none"}:
        return "-"
    parsed = datetime.fromisoformat(str(value))
    return parsed.strftime("%d %b, %H:%M" if short else "%d %b %Y, %H:%M")


def session_label(session: dict) -> str:
    live = "LIVE" if session.get("active") else "CLOSED"
    return f"#{session['session_id']} · {session['course_code']} · {fmt_datetime(session['starts_at'], short=True)} · {live}"


def navigate_to(page: str) -> None:
    st.session_state["navigation"] = page


def render_session_strip(session: dict) -> None:
    st.markdown(
        f"""
        <div class="session-strip">
            <div><div class="strip-label">Live class</div><div class="strip-value">{session['course_code']} / {session['course_name']}</div></div>
            <div><div class="strip-label">Venue</div><div class="strip-value">{session['venue']}</div></div>
            <div><div class="strip-label">Started</div><div class="strip-value">{fmt_datetime(session['starts_at'], short=True)}</div></div>
            <div class="head-badge"><span class="live-dot"></span>{session['attendance_count']} verified</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quality_label(score: float) -> str:
    if score >= 0.67:
        return "Good"
    if score >= 0.42:
        return "Usable"
    return "Low"


def pipeline_payload(result: PipelineResult) -> dict:
    """Return the JSON-safe enhancement record stored with an enrolment."""

    return {
        "schema_version": fingerprint_module.PIPELINE_SCHEMA_VERSION,
        "stages": result.stages,
        "quality": result.quality,
    }


def results_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    rename = {
        "student_id": "Student ID",
        "full_name": "Student",
        "programme": "Programme",
        "study_year": "Year",
        "tutorial_group": "Group",
        "course_code": "Course",
        "course_name": "Course name",
        "venue": "Venue",
        "marked_at": "Marked at",
        "attendance_status": "Status",
        "similarity": "Match",
        "capture_quality": "Quality",
        "processing_ms": "Processing (ms)",
    }
    frame = frame.rename(columns=rename)
    if "Marked at" in frame:
        frame["Marked at"] = frame["Marked at"].map(fmt_datetime)
    if "Match" in frame:
        frame["Match"] = frame["Match"].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "-")
    if "Quality" in frame:
        frame["Quality"] = frame["Quality"].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "-")
    if "Processing (ms)" in frame:
        frame["Processing (ms)"] = frame["Processing (ms)"].map(lambda value: f"{float(value):.0f}")
    preferred = [
        "Student ID",
        "Student",
        "Programme",
        "Year",
        "Group",
        "Course",
        "Status",
        "Marked at",
        "Match",
        "Quality",
        "Processing (ms)",
    ]
    return frame[[column for column in preferred if column in frame.columns]]


def load_demo_cohort() -> tuple[int, int, int]:
    """Install an explicitly labelled local demo cohort and one live session."""

    DEMO_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    cohort = [
        ("ST2026001", "Aisyah Rahman", PROGRAMMES[0], 3, "T6", 101),
        ("ST2026002", "Daniel Lee", PROGRAMMES[0], 3, "T6", 202),
        ("ST2026003", "Kavitha Nair", PROGRAMMES[1], 2, "T2", 303),
        ("ST2026004", "Muhammad Faris", PROGRAMMES[2], 1, "T4", 404),
    ]
    added = 0
    upgraded = 0
    for student_id, name, programme, year, group, seed in cohort:
        student_exists = get_student(student_id) is not None
        student_templates = [
            template
            for template in list_templates()
            if template["student_id"] == student_id
        ]
        has_reference = any(template.get("reference_path") for template in student_templates)
        demo_scan_path = DEMO_SCAN_DIR / f"{student_id}_{name.replace(' ', '_')}.png"
        if student_exists and has_reference and demo_scan_path.exists():
            continue
        base = generate_demo_fingerprint(seed)
        result = process_fingerprint(base)
        template_record = None
        if not student_exists or not has_reference:
            template_path = TEMPLATE_DIR / f"{student_id.lower()}_{uuid.uuid4().hex[:10]}.png"
            reference_path = REFERENCE_DIR / f"{student_id.lower()}_{uuid.uuid4().hex[:10]}_reference.png"
            save_enhanced_template(result, template_path)
            save_reference_capture(result, reference_path)
            template_record = {
                "finger_label": "Right index",
                "image_path": str(template_path.resolve()),
                "reference_path": str(reference_path.resolve()),
                "quality": result.quality["overall"],
                "clarity": result.quality["clarity"],
                "minutiae_count": int(result.quality.get("minutiae_count", 0)),
                "profile": pipeline_payload(result),
            }

        # A small rotation/noise difference makes the sample a realistic query,
        # while remaining deterministic for demonstrations and automated tests.
        matrix = cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), 2.0, 1.0)
        query = cv2.warpAffine(base, matrix, (base.shape[1], base.shape[0]), borderValue=245)
        noise = np.random.default_rng(seed + 1).normal(0, 2.2, query.shape)
        query = np.clip(query.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(demo_scan_path), query)

        if student_exists:
            if not has_reference and template_record is not None:
                add_fingerprint_templates(student_id, [template_record])
                upgraded += 1
        elif template_record is not None:
            enrol_student(
                {
                    "student_id": student_id,
                    "full_name": f"{name} (Demo)",
                    "programme": programme,
                    "study_year": year,
                    "tutorial_group": group,
                    "email": f"{student_id.lower()}@student.demo",
                },
                [template_record],
            )
            added += 1

    sessions = list_sessions()
    created = 0
    if not sessions:
        create_session(
            {
                "course_code": "BMDS2133",
                "course_name": "Image Processing",
                "venue": "Lab B-05-01",
                "lecturer": "Demo Lecturer",
                "starts_at": local_now() - timedelta(minutes=5),
                "grace_minutes": 15,
            }
        )
        created = 1
    return added, upgraded, created


def page_dashboard() -> None:
    page_header(
        "Operations overview",
        "Attendance dashboard",
        "A live view of biometric enrolment, open classes and verified attendance activity.",
        "LOCAL DATABASE",
    )
    statistics = dashboard_statistics()
    students = list_students()
    active_sessions = list_sessions(active_only=True)
    recent = attendance_records()[:8]

    columns = st.columns(4)
    with columns[0]:
        stat_card("Enrolled students", str(statistics["students"]), "Active biometric identities")
    with columns[1]:
        stat_card("Live sessions", str(statistics["active_sessions"]), "Accepting attendance scans")
    with columns[2]:
        stat_card("Verified today", str(statistics["attendance_today"]), "Present and late records")
    with columns[3]:
        average = statistics["average_similarity"]
        stat_card("Average match", f"{average:.1%}" if average else "-", "Across recorded decisions")

    st.subheader("Open classes")
    if active_sessions:
        for session in active_sessions[:3]:
            render_session_strip(session)
        st.button(
            "Open attendance scanner",
            type="primary",
            on_click=navigate_to,
            args=("Mark attendance",),
        )
    else:
        st.info("No class session is open. Create one before accepting fingerprint scans.")
        st.button("Create a class session", on_click=navigate_to, args=("Class sessions",))

    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.subheader("Recent verifications")
        if recent:
            st.dataframe(results_dataframe(recent), width="stretch", hide_index=True)
        else:
            st.caption("Verified attendance will appear here after the first successful scan.")
    with right:
        st.subheader("Cohort coverage")
        if students:
            frame = pd.DataFrame(students)
            programme_counts = frame.groupby("programme")["student_id"].count().sort_values(ascending=False)
            st.bar_chart(programme_counts)
        else:
            st.caption("Enrol students or load the labelled demo cohort from System & help.")

    st.subheader("How a scan becomes attendance")
    pipeline_strip()


def render_scan_result(payload: dict) -> None:
    result: PipelineResult = payload["pipeline"]
    identification: IdentificationResult = payload["identification"]

    if identification.matched and identification.student:
        student = identification.student
        if payload["created"]:
            result_banner(
                "success",
                f"Attendance marked - {student['full_name']}",
                f"{student['student_id']} was verified as {payload['status']} for {payload['session_label']}.",
            )
        else:
            result_banner(
                "warning",
                f"Already recorded - {student['full_name']}",
                f"Duplicate prevention retained the original {payload['status']} record from {fmt_datetime(payload['record']['marked_at'])}.",
            )
    else:
        result_banner("fail", "Fingerprint not accepted", identification.reason)

    metrics = st.columns(4)
    similarity = identification.evidence.similarity if identification.evidence else 0.0
    with metrics[0]:
        st.metric("Match score", f"{similarity:.1%}")
    with metrics[1]:
        st.metric("Capture quality", f"{result.quality['overall']:.1%}", quality_label(result.quality["overall"]))
    with metrics[2]:
        st.metric("Ridge continuity", f"{result.quality['connectivity']:.1%}", "After post-processing")
    with metrics[3]:
        st.metric("Processing", f"{payload['elapsed_ms']:.0f} ms", f"{identification.candidates_checked} templates")

    st.subheader("Fingerprint identity comparison")
    reference_path = identification.reference_path or identification.template_path
    reference_image = (
        cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference_path
        else None
    )
    comparison_columns = st.columns(2, gap="large")
    with comparison_columns[0]:
        st.image(
            result.prepared,
            caption="Current attendance capture — uploaded for this scan",
            width="stretch",
            clamp=True,
        )
    with comparison_columns[1]:
        if reference_image is not None:
            reference_caption = (
                "Matched database enrolment — verified reference"
                if identification.matched
                else "Closest database reference — not accepted"
            )
            st.image(reference_image, caption=reference_caption, width="stretch", clamp=True)
        else:
            st.info("No database fingerprint reached the comparison stage.")

    if identification.matched and identification.student:
        student = identification.student
        identity_columns = st.columns(4)
        identity_columns[0].metric("Registered student", str(student.get("full_name", "-")))
        identity_columns[1].metric("Student ID", str(student.get("student_id", "-")))
        identity_columns[2].metric("Enrolled finger", str(student.get("finger_label", "-")))
        identity_columns[3].metric(
            "Programme / group",
            str(student.get("tutorial_group", "-")),
            str(student.get("programme", "-")),
        )
        inliers = identification.evidence.reference_inliers if identification.evidence else 0
        st.success(
            f"The current capture was linked to {student.get('full_name', 'this student')} "
            f"using {inliers} geometrically consistent local fingerprint features."
        )
    elif reference_image is not None:
        st.warning(
            "The right image is only the closest database candidate. Its identity was not accepted, "
            "so no student metadata or attendance record was assigned."
        )

    with st.expander("Technical decision evidence"):
        if identification.evidence:
            evidence = identification.evidence
            st.markdown("#### Biometric feature comparison")
            st.caption(
                "These are fingerprint features derived from the image—not filename, EXIF or other file metadata. "
                "The filters prepare the ridges; the matcher then compares local descriptors and ridge structure."
            )
            feature_comparison = pd.DataFrame(
                [
                    {
                        "Evidence": "SIFT local ridge features",
                        "Current capture": f"{evidence.keypoints_query:,} keypoints",
                        "Enrolled reference": f"{evidence.keypoints_template:,} keypoints",
                        "Comparison": f"{evidence.reference_matches} mutual pairs",
                        "Result": f"{evidence.reference_inliers} RANSAC inliers",
                    },
                    {
                        "Evidence": "Reference geometry",
                        "Current capture": "Local descriptors",
                        "Enrolled reference": "Local descriptors",
                        "Comparison": "Translation / rotation / scale consistency",
                        "Result": f"{evidence.reference_score:.1%}",
                    },
                    {
                        "Evidence": "Enhanced ORB geometry",
                        "Current capture": "Filtered binary ridges",
                        "Enrolled reference": "Stored enhanced template",
                        "Comparison": f"{evidence.good_matches} candidate pairs",
                        "Result": f"{evidence.orb_score:.1%} ({evidence.geometric_inliers} inliers)",
                    },
                    {
                        "Evidence": "Aligned ridge structure",
                        "Current capture": "Ridge direction field",
                        "Enrolled reference": "Ridge direction field",
                        "Comparison": "Aligned flow agreement",
                        "Result": f"{evidence.structural_score:.1%}",
                    },
                    {
                        "Evidence": "Local ridge spectrum",
                        "Current capture": "Frequency descriptor",
                        "Enrolled reference": "Frequency descriptor",
                        "Comparison": "Spectral similarity",
                        "Result": f"{evidence.spectral_score:.1%}",
                    },
                ]
            )
            st.dataframe(feature_comparison, width="stretch", hide_index=True)

            threshold_used = float(payload.get("threshold", DEFAULT_MATCH_THRESHOLD))
            ambiguity_used = float(payload.get("ambiguity_margin", DEFAULT_AMBIGUITY_MARGIN))
            lead = max(similarity - identification.runner_up_similarity, 0.0)
            decision_columns = st.columns(4)
            decision_columns[0].metric("Fused match score", f"{similarity:.1%}")
            decision_columns[1].metric("Required score", f"{threshold_used:.1%}")
            decision_columns[2].metric("Lead over runner-up", f"{lead:.1%}")
            decision_columns[3].metric("Required lead", f"{ambiguity_used:.1%}")
            st.caption(
                "Score fusion: 54% reference geometry + 18% enhanced ORB geometry + "
                "20% aligned ridge structure + 8% local ridge spectrum."
            )
            reference_mode = "canonical enrolment upload" if evidence.used_canonical_reference else "legacy enhanced-template fallback"
            st.caption(f"Primary comparison source: {reference_mode}.")

        st.markdown("#### Filter execution on the current attendance capture")
        st.caption(f"Executed enhancement: {' -> '.join(result.stages)}.")
        diagnostic_columns = st.columns(6)
        with diagnostic_columns[0]:
            st.image(result.contrast_enhanced, caption="1. CLAHE local contrast enhancement", width="stretch")
        with diagnostic_columns[1]:
            st.image(result.denoised, caption="2. Bilateral edge-preserving denoising", width="stretch")
        with diagnostic_columns[2]:
            st.image(result.detail_enhanced, caption="3. Mild unsharp ridge enhancement", width="stretch")
        with diagnostic_columns[3]:
            st.image(result.binary, caption="4. Adaptive local-mean binary", width="stretch")
        with diagnostic_columns[4]:
            st.image(result.thinned, caption="5. Morphological thinning", width="stretch")
        with diagnostic_columns[5]:
            st.image(result.enhanced, caption="6. Binary ridge post-processing", width="stretch")
        st.image(
            result.region_mask,
            caption="Texture-based fingertip foreground mask",
            width=260,
        )
        st.caption(
            "Acceptance requires the configured match threshold and a clear lead over the runner-up. "
            "The score supports a classroom prototype and is not a forensic biometric certification."
        )


def page_mark_attendance() -> None:
    page_header(
        "Biometric verification",
        "Mark attendance",
        "Select a live class, provide a fresh fingerprint capture and let the ridge-preserving enhancement pipeline identify the enrolled student.",
        "SCANNER READY",
    )
    sessions = list_sessions(active_only=True)
    if not sessions:
        st.warning("Attendance cannot be marked until a class session is open.")
        st.button("Create a class session", type="primary", on_click=navigate_to, args=("Class sessions",))
        return

    session_map = {session_label(session): session for session in sessions}
    selected_label = st.selectbox("Open class session", list(session_map))
    selected_session = session_map[selected_label]
    render_session_strip(selected_session)

    with st.expander("Decision controls", expanded=False):
        threshold = st.slider(
            "Acceptance threshold",
            min_value=0.45,
            max_value=0.85,
            value=float(DEFAULT_MATCH_THRESHOLD),
            step=0.01,
            help="Higher values reduce false acceptance but may reject more genuine scans.",
        )
        ambiguity = st.slider(
            "Minimum lead over runner-up",
            min_value=0.01,
            max_value=0.12,
            value=float(DEFAULT_AMBIGUITY_MARGIN),
            step=0.005,
        )

    source_mode = st.radio(
        "Fingerprint source",
        ["Upload fingerprint photo", "Live camera capture"],
        horizontal=True,
    )
    source = None
    automatic_camera_verification = False
    camera_digest = None
    if source_mode == "Upload fingerprint photo":
        uploaded = st.file_uploader(
            "Drop a fingerprint capture",
            type=IMAGE_TYPES,
            accept_multiple_files=False,
            help="PNG, JPG, BMP or TIFF. Greyscale conversion is automatic.",
        )
        if uploaded:
            source = uploaded.getvalue()
    elif source_mode == "Live camera capture":
        st.info(
            "Place the enrolled fingertip in the centre, fill most of the frame and keep the ridges in focus. "
            "After you press the camera's capture button, verification starts automatically."
        )
        camera_capture = st.camera_input(
            "Camera fingerprint capture",
            key="attendance_camera",
            help="The browser may ask for camera permission. Camera access stays on this local app.",
        )
        if camera_capture:
            source = camera_capture.getvalue()
            camera_digest = sha256(source).hexdigest()
            automatic_camera_verification = (
                st.session_state.get("last_live_camera_digest") != camera_digest
            )
            if not automatic_camera_verification:
                st.caption("This camera image has already been checked. Retake the photo to run a new scan.")
    manual_verification = False
    if source_mode != "Live camera capture":
        manual_verification = st.button(
            "Enhance, verify and mark attendance",
            type="primary",
            width="stretch",
        )

    if manual_verification or automatic_camera_verification:
        if source is None:
            st.warning("Provide a fingerprint capture first.")
        elif not list_templates():
            st.warning("No enrolled fingerprint templates are available.")
        else:
            if camera_digest:
                st.session_state["last_live_camera_digest"] = camera_digest
            try:
                started = perf_counter()
                with st.spinner("Enhancing ridges and comparing enrolled templates..."):
                    pipeline = process_fingerprint(source)
                    identification = identify_student(
                        pipeline,
                        list_templates(),
                        threshold=threshold,
                        ambiguity_margin=ambiguity,
                    )
                    created = False
                    record = {}
                    status = "Not recorded"
                    if identification.matched and identification.student:
                        status = attendance_status(selected_session)
                        created, record = mark_attendance(
                            selected_session["session_id"],
                            identification.student["student_id"],
                            status,
                            identification.evidence.similarity if identification.evidence else 0.0,
                            pipeline.quality["overall"],
                            pipeline.processing_ms,
                        )
                        if not created:
                            status = record["attendance_status"]
                st.session_state["last_scan"] = {
                    "pipeline": pipeline,
                    "identification": identification,
                    "created": created,
                    "record": record,
                    "status": status,
                    "session_label": f"{selected_session['course_code']} / {selected_session['course_name']}",
                    "elapsed_ms": (perf_counter() - started) * 1000.0,
                    "threshold": threshold,
                    "ambiguity_margin": ambiguity,
                }
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The fingerprint could not be processed: {exc}")

    if "last_scan" in st.session_state:
        st.divider()
        render_scan_result(st.session_state["last_scan"])


def page_enrolment() -> None:
    page_header(
        "Identity administration",
        "Student enrolment",
        "Register student details with one to three captures of the same finger. Each upload becomes a local matching reference.",
        "LOCAL ENROLMENT",
    )

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.subheader("New biometric identity")
        with st.form("enrolment_form", clear_on_submit=False):
            first, second = st.columns(2)
            student_id = first.text_input("Student ID", placeholder="e.g. 2510012")
            full_name = second.text_input("Full name", placeholder="As shown in university records")
            programme = st.selectbox("Programme", PROGRAMMES)
            first, second = st.columns(2)
            study_year = first.number_input("Year of study", min_value=1, max_value=8, value=1, step=1)
            tutorial_group = second.text_input("Tutorial group", placeholder="e.g. T6")
            email = st.text_input("Student email (optional)", placeholder="student@university.edu.my")
            finger_label = st.selectbox(
                "Enrolled finger",
                ["Right index", "Left index", "Right thumb", "Left thumb", "Other"],
            )
            uploads = st.file_uploader(
                "Fingerprint captures (1-3 of the same finger)",
                type=IMAGE_TYPES,
                accept_multiple_files=True,
            )
            submitted = st.form_submit_button("Process and enrol student", type="primary", width="stretch")

        if submitted:
            errors = []
            if not student_id.strip():
                errors.append("Student ID is required.")
            if not full_name.strip():
                errors.append("Full name is required.")
            if not tutorial_group.strip():
                errors.append("Tutorial group is required.")
            if not uploads:
                errors.append("At least one fingerprint capture is required.")
            if uploads and len(uploads) > 3:
                errors.append("Use no more than three captures during enrolment.")
            if get_student(student_id) if student_id.strip() else False:
                errors.append("That student ID is already enrolled.")

            if errors:
                for message in errors:
                    st.error(message)
            else:
                stored_paths: list[Path] = []
                try:
                    template_rows = []
                    preview_results = []
                    with st.spinner("Running ridge-preserving enhancement and quality checks..."):
                        for upload in uploads:
                            result = process_fingerprint(upload.getvalue())
                            quality_issue = capture_quality_issue(result)
                            if quality_issue:
                                raise ValueError(f"{upload.name}: {quality_issue}")
                            capture_key = f"{student_id.strip().lower()}_{uuid.uuid4().hex[:12]}"
                            path = TEMPLATE_DIR / f"{capture_key}_enhanced.png"
                            reference_path = REFERENCE_DIR / f"{capture_key}_reference.png"
                            save_enhanced_template(result, path)
                            stored_paths.append(path)
                            save_reference_capture(result, reference_path)
                            stored_paths.append(reference_path)
                            template_rows.append(
                                {
                                    "finger_label": finger_label,
                                    "image_path": str(path.resolve()),
                                    "reference_path": str(reference_path.resolve()),
                                    "quality": result.quality["overall"],
                                    "clarity": result.quality["clarity"],
                                    "minutiae_count": int(result.quality.get("minutiae_count", 0)),
                                    "profile": pipeline_payload(result),
                                }
                            )
                            preview_results.append(result)
                        enrol_student(
                            {
                                "student_id": student_id,
                                "full_name": full_name,
                                "programme": programme,
                                "study_year": int(study_year),
                                "tutorial_group": tutorial_group,
                                "email": email,
                            },
                            template_rows,
                        )
                    st.session_state["enrolment_preview"] = preview_results
                    st.success(f"{full_name} was enrolled with {len(template_rows)} fingerprint template(s).")
                except (ValueError, sqlite3.IntegrityError) as exc:
                    for path in stored_paths:
                        path.unlink(missing_ok=True)
                    st.error(str(exc))
                except Exception as exc:
                    for path in stored_paths:
                        path.unlink(missing_ok=True)
                    st.error(f"Enrolment could not be completed: {exc}")

        if "enrolment_preview" in st.session_state:
            preview: PipelineResult = st.session_state["enrolment_preview"][0]
            st.subheader("Stored biometric reference")
            preview_columns = st.columns([1.05, 1], gap="large")
            preview_columns[0].image(
                preview.prepared,
                caption="Canonical fingerprint reference saved for future attendance matching",
                width="stretch",
            )
            with preview_columns[1]:
                st.markdown("#### Extracted biometric profile")
                profile_columns = st.columns(2)
                profile_columns[0].metric("Capture quality", f"{preview.quality['overall']:.1%}")
                profile_columns[1].metric("Ridge coherence", f"{preview.quality['ridge_coherence']:.1%}")
                profile_columns[0].metric("Ridge continuity", f"{preview.quality['connectivity']:.1%}")
                profile_columns[1].metric("Detected ridge points", f"{int(preview.quality.get('minutiae_count', 0)):,}")
                st.info(
                    "The system stores this normalized reference and a processed ridge template. "
                    "Student information links those biometric records to the enrolled identity."
                )
            with st.expander("Show enrolment filter evidence"):
                filter_columns = st.columns(3)
                filter_columns[0].image(preview.contrast_enhanced, caption="1. CLAHE contrast", width="stretch")
                filter_columns[1].image(preview.detail_enhanced, caption="3. Ridge detail enhancement", width="stretch")
                filter_columns[2].image(preview.enhanced, caption="6. Stored binary template", width="stretch")
                st.caption(f"Executed enhancement: {' -> '.join(preview.stages)}.")

    with right:
        st.subheader("Enrolment quality guide")
        pipeline_strip()
        st.info(
            "For the most reliable verification, capture the same finger two or three times with "
            "steady pressure, a clean sensor and the full fingertip inside the frame."
        )
        st.markdown(
            """
            **Capture checklist**

            - Use a sharp image with visible ridge/valley contrast.
            - Centre one upright fingertip against a plain, contrasting background.
            - Use the 1x or macro camera, not 0.5x ultrawide; let the fingertip fill 60-80% of the frame.
            - Avoid motion blur, heavy moisture and clipped fingertip edges.
            - Keep every enrolment sample from the same finger.
            - Recapture when quality is labelled low.
            """
        )

    existing_students = list_students()
    if existing_students:
        with st.expander("Upgrade an existing student's fingerprint reference"):
            st.caption(
                "Use this for students enrolled before canonical reference storage was introduced, "
                "or to add another capture of the same finger. Existing records are preserved."
            )
            student_map = {
                f"{student['student_id']} - {student['full_name']}": student
                for student in existing_students
            }
            selected_student_label = st.selectbox(
                "Existing student",
                list(student_map),
                key="existing_fingerprint_student",
            )
            existing_finger = st.selectbox(
                "Finger used for these captures",
                ["Right index", "Left index", "Right thumb", "Left thumb", "Other"],
                key="existing_fingerprint_label",
            )
            existing_uploads = st.file_uploader(
                "New fingerprint captures (1-3 of the same finger)",
                type=IMAGE_TYPES,
                accept_multiple_files=True,
                key="existing_fingerprint_uploads",
            )
            if st.button("Process and add canonical references", type="primary"):
                if not existing_uploads:
                    st.warning("Upload at least one new fingerprint capture.")
                elif len(existing_uploads) > 3:
                    st.warning("Use no more than three captures in one update.")
                else:
                    stored_paths: list[Path] = []
                    try:
                        selected_student = student_map[selected_student_label]
                        template_rows = []
                        previews = []
                        with st.spinner("Creating current-format reference captures..."):
                            for upload in existing_uploads:
                                result = process_fingerprint(upload.getvalue())
                                quality_issue = capture_quality_issue(result)
                                if quality_issue:
                                    raise ValueError(f"{upload.name}: {quality_issue}")
                                capture_key = (
                                    f"{selected_student['student_id'].lower()}_"
                                    f"{uuid.uuid4().hex[:12]}"
                                )
                                template_path = TEMPLATE_DIR / f"{capture_key}_enhanced.png"
                                reference_path = REFERENCE_DIR / f"{capture_key}_reference.png"
                                save_enhanced_template(result, template_path)
                                save_reference_capture(result, reference_path)
                                stored_paths.extend([template_path, reference_path])
                                template_rows.append(
                                    {
                                        "finger_label": existing_finger,
                                        "image_path": str(template_path.resolve()),
                                        "reference_path": str(reference_path.resolve()),
                                        "quality": result.quality["overall"],
                                        "clarity": result.quality["clarity"],
                                        "minutiae_count": int(result.quality.get("minutiae_count", 0)),
                                        "profile": pipeline_payload(result),
                                    }
                                )
                                previews.append(result)
                            add_fingerprint_templates(
                                selected_student["student_id"],
                                template_rows,
                            )
                        st.session_state["enrolment_preview"] = previews
                        st.success(
                            f"Added {len(template_rows)} canonical reference capture(s) for "
                            f"{selected_student['full_name']}."
                        )
                    except Exception as exc:
                        for path in stored_paths:
                            path.unlink(missing_ok=True)
                        st.error(f"The fingerprint reference could not be added: {exc}")

    st.divider()
    st.subheader("Student directory")
    students = list_students()
    if students:
        directory = pd.DataFrame(students).rename(
            columns={
                "student_id": "Student ID",
                "full_name": "Student",
                "programme": "Programme",
                "study_year": "Year",
                "tutorial_group": "Group",
                "templates": "Templates",
                "average_quality": "Average quality",
                "status": "Status",
            }
        )
        directory["Average quality"] = directory["Average quality"].map(
            lambda value: f"{float(value):.1%}" if pd.notna(value) else "-"
        )
        st.dataframe(
            directory[["Student ID", "Student", "Programme", "Year", "Group", "Templates", "Average quality", "Status"]],
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Manage a student")
        student_map = {
            f"{student['student_id']} · {student['full_name']}": student
            for student in students
        }
        managed_label = st.selectbox("Student record", list(student_map), key="managed_student")
        managed = student_map[managed_label]
        edit_tab, delete_tab = st.tabs(["Edit details", "Delete record"])
        with edit_tab:
            with st.form(f"edit_student_{managed['student_id']}"):
                st.text_input("Student ID", value=managed["student_id"], disabled=True)
                edited_name = st.text_input("Full name", value=managed["full_name"])
                programme_options = list(PROGRAMMES)
                if managed["programme"] not in programme_options:
                    programme_options.append(managed["programme"])
                edited_programme = st.selectbox(
                    "Programme",
                    programme_options,
                    index=programme_options.index(managed["programme"]),
                )
                edit_left, edit_right = st.columns(2)
                edited_year = edit_left.number_input(
                    "Year of study", min_value=1, max_value=8, value=int(managed["study_year"])
                )
                edited_group = edit_right.text_input("Tutorial group", value=managed["tutorial_group"])
                edited_email = st.text_input("Email", value=managed.get("email", ""))
                edited_status = st.selectbox(
                    "Status",
                    ["Active", "Inactive"],
                    index=0 if managed.get("status") == "Active" else 1,
                    help="Inactive students remain in the directory but are excluded from matching and class rosters.",
                )
                save_student = st.form_submit_button("Save student changes", type="primary", width="stretch")
            if save_student:
                try:
                    update_student(
                        managed["student_id"],
                        {
                            "full_name": edited_name,
                            "programme": edited_programme,
                            "study_year": int(edited_year),
                            "tutorial_group": edited_group,
                            "email": edited_email,
                            "status": edited_status,
                        },
                    )
                    st.success("Student details updated.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with delete_tab:
            st.warning(
                "Deleting this student also removes their fingerprint templates and attendance entries. "
                "This cannot be undone."
            )
            confirm_student_delete = st.checkbox(
                f"I understand and want to delete {managed['student_id']}",
                key=f"confirm_student_delete_{managed['student_id']}",
            )
            if st.button(
                "Delete student permanently",
                key=f"delete_student_{managed['student_id']}",
                disabled=not confirm_student_delete,
                width="stretch",
            ):
                removed_files = delete_student(managed["student_id"])
                st.session_state.pop("last_scan", None)
                st.success(f"Student deleted. {removed_files} local fingerprint file(s) removed.")
                st.rerun()
    else:
        st.caption("No students are enrolled yet.")


def page_sessions() -> None:
    page_header(
        "Class administration",
        "Class sessions",
        "Open a scheduled class for fingerprint attendance, configure its grace period and close it when teaching ends.",
        "SESSION CONTROL",
    )
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Open a new session")
        with st.form("session_form", clear_on_submit=True):
            course_code = st.text_input("Course code", placeholder="BMDS2133")
            course_name = st.text_input("Course name", placeholder="Image Processing")
            venue = st.text_input("Venue", placeholder="Lab B-05-01")
            lecturer = st.text_input("Lecturer", placeholder="Lecturer name")
            first, second = st.columns(2)
            session_date = first.date_input("Date", value=local_now().date())
            session_time = second.time_input("Start time", value=local_now().replace(second=0, microsecond=0).time())
            grace_minutes = st.number_input("On-time grace period (minutes)", min_value=0, max_value=60, value=15, step=5)
            submitted = st.form_submit_button("Open class session", type="primary", width="stretch")
        if submitted:
            if not all(value.strip() for value in (course_code, course_name, venue, lecturer)):
                st.error("Course, venue and lecturer fields are required.")
            else:
                starts_at = datetime.combine(session_date, session_time).astimezone()
                create_session(
                    {
                        "course_code": course_code,
                        "course_name": course_name,
                        "venue": venue,
                        "lecturer": lecturer,
                        "starts_at": starts_at,
                        "grace_minutes": int(grace_minutes),
                    }
                )
                st.success(f"{course_code.upper()} is now accepting attendance scans.")
                st.rerun()

    with right:
        st.subheader("Open now")
        active = list_sessions(active_only=True)
        if not active:
            st.caption("No sessions are currently open.")
        for session in active:
            render_session_strip(session)
        if active:
            st.caption("Use Manage a session below to edit, close, reopen or delete a class.")

    st.divider()
    st.subheader("Session history")
    sessions = list_sessions()
    if sessions:
        frame = pd.DataFrame(sessions)
        frame["Status"] = frame["active"].map({1: "Open", 0: "Closed"})
        frame["Start"] = frame["starts_at"].map(fmt_datetime)
        frame["End"] = frame["ends_at"].map(fmt_datetime)
        frame = frame.rename(
            columns={
                "course_code": "Course",
                "course_name": "Course name",
                "venue": "Venue",
                "lecturer": "Lecturer",
                "attendance_count": "Attendance",
                "late_count": "Late",
            }
        )
        st.dataframe(
            frame[["Course", "Course name", "Start", "End", "Venue", "Lecturer", "Attendance", "Late", "Status"]],
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Manage a session")
        session_map = {session_label(session): session for session in sessions}
        managed_label = st.selectbox("Class session record", list(session_map), key="managed_session")
        managed = session_map[managed_label]
        managed_start = datetime.fromisoformat(managed["starts_at"])
        edit_tab, delete_tab = st.tabs(["Edit session", "Delete session"])
        with edit_tab:
            with st.form(f"edit_session_{managed['session_id']}"):
                edit_code = st.text_input("Course code", value=managed["course_code"])
                edit_name = st.text_input("Course name", value=managed["course_name"])
                edit_venue = st.text_input("Venue", value=managed["venue"])
                edit_lecturer = st.text_input("Lecturer", value=managed["lecturer"])
                date_column, time_column = st.columns(2)
                edit_date = date_column.date_input("Date", value=managed_start.date())
                edit_time = time_column.time_input("Start time", value=managed_start.time())
                edit_grace = st.number_input(
                    "On-time grace period (minutes)",
                    min_value=0,
                    max_value=60,
                    value=int(managed["grace_minutes"]),
                    step=5,
                )
                edit_active = st.checkbox(
                    "Session is open for attendance",
                    value=bool(managed["active"]),
                    help="Clear this to close the session. Select it again later to reopen the session.",
                )
                save_session = st.form_submit_button("Save session changes", type="primary", width="stretch")
            if save_session:
                if not all(value.strip() for value in (edit_code, edit_name, edit_venue, edit_lecturer)):
                    st.error("Course, venue and lecturer fields are required.")
                else:
                    starts_at = datetime.combine(edit_date, edit_time).astimezone()
                    update_session(
                        managed["session_id"],
                        {
                            "course_code": edit_code,
                            "course_name": edit_name,
                            "venue": edit_venue,
                            "lecturer": edit_lecturer,
                            "starts_at": starts_at,
                            "grace_minutes": int(edit_grace),
                            "active": edit_active,
                            "ends_at": managed.get("ends_at"),
                        },
                    )
                    st.success("Session updated.")
                    st.rerun()
        with delete_tab:
            st.warning(
                f"Deleting session #{managed['session_id']} also removes its "
                f"{managed['attendance_count']} attendance record(s). This cannot be undone."
            )
            confirm_session_delete = st.checkbox(
                f"I understand and want to delete session #{managed['session_id']}",
                key=f"confirm_session_delete_{managed['session_id']}",
            )
            if st.button(
                "Delete session permanently",
                key=f"delete_session_{managed['session_id']}",
                disabled=not confirm_session_delete,
                width="stretch",
            ):
                delete_session(managed["session_id"])
                st.session_state.pop("last_scan", None)
                st.success("Session and its attendance records were deleted.")
                st.rerun()


def page_records() -> None:
    page_header(
        "Evidence and export",
        "Attendance records",
        "Review verified scans, inspect complete class rosters and export records for academic administration.",
        "AUDITABLE RECORDS",
    )
    sessions = list_sessions()
    if not sessions:
        st.info("Create a class session before attendance records can be produced.")
        return

    session_map = {session_label(session): session for session in sessions}
    with st.container(border=True):
        st.markdown("#### Choose a class session")
        filter_columns = st.columns([1.45, 1])
        chosen = filter_columns[0].selectbox("Class session", list(session_map))
        search = filter_columns[1].text_input(
            "Search within this session", placeholder="Student name, ID or programme"
        )
    session = session_map[chosen]
    selected_id = int(session["session_id"])

    roster = session_roster(selected_id)
    if search.strip():
        term = search.casefold()
        roster = [
            row
            for row in roster
            if term in row["student_id"].casefold()
            or term in row["full_name"].casefold()
            or term in row["programme"].casefold()
        ]
    present = sum(row["attendance_status"] == "Present" for row in roster)
    late = sum(row["attendance_status"] == "Late" for row in roster)
    absent = sum(row["attendance_status"] == "Absent" for row in roster)
    rate = (present + late) / len(roster) if roster else 0

    st.markdown('<div class="ui-spacer"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"### Session #{selected_id}: {session['course_code']} / {session['course_name']}")
        st.caption(
            f"{fmt_datetime(session['starts_at'])} · {session['venue']} · {session['lecturer']} · "
            f"{'Open for attendance' if session['active'] else 'Closed session'}"
        )
        stats = st.columns(4)
        with stats[0]:
            stat_card("Session roster", str(len(roster)), f"Session #{selected_id}")
        with stats[1]:
            stat_card("Present", str(present), "Within the grace period")
        with stats[2]:
            stat_card("Late", str(late), "After the grace period")
        with stats[3]:
            stat_card("Attendance rate", f"{rate:.0%}", f"{absent} absent")
        st.caption(
            "A database uniqueness rule permits only one attendance entry for each student in this session."
        )

    frame = pd.DataFrame(roster)
    if not frame.empty:
        display = frame.rename(
            columns={
                "student_id": "Student ID",
                "full_name": "Student",
                "programme": "Programme",
                "study_year": "Year",
                "tutorial_group": "Group",
                "attendance_status": "Status",
                "marked_at": "Marked at",
                "similarity": "Match",
                "capture_quality": "Quality",
            }
        )
        display.insert(0, "Session", f"#{selected_id} · {session['course_code']}")
        display["Marked at"] = display["Marked at"].map(fmt_datetime)
        display["Match"] = display["Match"].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "-")
        display["Quality"] = display["Quality"].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "-")
        st.markdown('<div class="ui-spacer"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### Student attendance register")
            st.caption("Present, late and absent students for the selected session are listed together.")
            st.dataframe(
                display[["Session", "Student ID", "Student", "Programme", "Year", "Group", "Status", "Marked at", "Match", "Quality"]],
                width="stretch",
                hide_index=True,
            )
        csv_data = display.to_csv(index=False).encode("utf-8")
        pdf_bytes = build_attendance_pdf(session, roster)
        st.markdown('<div class="ui-spacer"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### Export this session")
            st.caption("Download the same selected-session register for spreadsheet review or printing.")
            export_columns = st.columns(2)
            export_columns[0].download_button(
                "Download CSV register",
                data=csv_data,
                file_name=f"session_{selected_id}_{session['course_code']}_attendance.csv",
                mime="text/csv",
                width="stretch",
            )
            export_columns[1].download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name=f"session_{selected_id}_{session['course_code']}_attendance.pdf",
                mime="application/pdf",
                width="stretch",
            )

    st.markdown('<div class="ui-spacer"></div>', unsafe_allow_html=True)
    with st.expander("All sessions overview"):
        overview = pd.DataFrame(
            [
                {
                    "Session": f"#{item['session_id']}",
                    "Course": item["course_code"],
                    "Course name": item["course_name"],
                    "Date": fmt_datetime(item["starts_at"]),
                    "Verified records": int(item["attendance_count"]),
                    "Late": int(item["late_count"] or 0),
                    "Status": "Open" if item["active"] else "Closed",
                }
                for item in sessions
            ]
        )
        st.dataframe(overview, width="stretch", hide_index=True)


def render_studio_result(result: PipelineResult) -> None:
    required_outputs = (
        "stages",
        "region_mask",
        "contrast_enhanced",
        "denoised",
        "detail_enhanced",
        "binary",
        "thinned",
        "enhanced",
    )
    if any(not hasattr(result, output) for output in required_outputs):
        st.session_state.pop("studio_result", None)
        st.warning(
            "The cached diagnostic result used an older processing pipeline. "
            "Upload the image and run the ridge-preserving enhancement pipeline again."
        )
        return
    st.info(f"Fixed executed sequence: **{' -> '.join(result.stages)}**.")
    st.subheader("Foreground extraction (executed before every filter)")
    foreground_columns = st.columns(3)
    foreground_columns[0].image(result.original, caption="Phone photograph", width="stretch")
    foreground_columns[1].image(result.prepared, caption="Cropped and normalized fingerprint ROI", width="stretch")
    foreground_columns[2].image(result.region_mask, caption="Final ridge foreground mask", width="stretch")
    metrics = st.columns(6)
    metrics[0].metric("Overall quality", f"{result.quality['overall']:.1%}")
    metrics[1].metric("Local contrast", f"{result.quality['contrast']:.1%}")
    metrics[2].metric("Noise suppression", f"{result.quality['noise_reduction']:.1%}")
    metrics[3].metric("Ridge continuity", f"{result.quality['connectivity']:.1%}")
    metrics[4].metric("Ridge coherence", f"{result.quality['ridge_coherence']:.1%}")
    metrics[5].metric("Runtime", f"{result.processing_ms:.0f} ms")

    stage_columns = st.columns(6)
    stage_columns[0].image(result.contrast_enhanced, caption="1. CLAHE local contrast enhancement", width="stretch")
    stage_columns[1].image(result.denoised, caption="2. Bilateral edge-preserving denoising", width="stretch")
    stage_columns[2].image(result.detail_enhanced, caption="3. Mild unsharp ridge enhancement", width="stretch")
    stage_columns[3].image(result.binary, caption="4. Local-mean binarization", width="stretch")
    stage_columns[4].image(result.thinned, caption="5. Morphological thinning", width="stretch")
    stage_columns[5].image(result.enhanced, caption="6. Binary ridge post-processing", width="stretch")


def page_algorithm_studio() -> None:
    page_header(
        "Image processing evidence",
        "Fingerprint enhancement studio",
        "Inspect each filtering result from local contrast enhancement to the final cleaned binary ridges.",
        "RIDGE-PRESERVING PIPELINE",
    )
    st.subheader("Fixed production pipeline")
    pipeline_strip()
    st.caption(
        "The numbered cards are the complete executed order. Matching remains a separate attendance operation after enhancement."
    )

    upload = st.file_uploader("Analyse a fingerprint image", type=IMAGE_TYPES, key="studio_upload")
    if st.button("Run live enhancement diagnostics", type="primary"):
        if not upload:
            st.warning("Upload a fingerprint image first.")
        else:
            try:
                with st.spinner("Running the complete ridge-preserving enhancement pipeline..."):
                    st.session_state["studio_result"] = process_fingerprint(upload.getvalue())
            except Exception as exc:
                st.error(f"The image could not be analysed: {exc}")
    if "studio_result" in st.session_state:
        render_studio_result(st.session_state["studio_result"])

    st.divider()
    st.subheader("Enhancement method comparison")
    st.caption(
        "The app uses the method that preserves visible phone-camera evidence. It does not invent accuracy values; "
        "submit quantitative claims only after running a labelled evaluation dataset."
    )
    comparison = pd.DataFrame(
        [
            ["CLAHE + bilateral + mild unsharp", "Local contrast + edge-preserving detail", "No", "Greyscale + binary", "Production pipeline"],
            ["STFT contextual filtering", "Local ridge frequency + orientation", "Yes", "Greyscale", "Rejected for current phone captures: synthetic coarse lines"],
            ["Modified Gabor", "Orientation + ridge frequency", "Yes", "Greyscale", "Future controlled-capture comparison"],
        ],
        columns=["Method", "Main information", "Frequency required", "Output", "Use here"],
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.info(
        "Do not report accuracy from this comparison table. Measure genuine and impostor captures on a labelled test set."
    )

    with st.expander("Implementation pseudocode"):
        st.code(
            """INPUT greyscale fingerprint capture
SYSTEM EXTENSION: locate fingerprint foreground and reject scanner borders
1. APPLY CLAHE local contrast enhancement with clip limit 2.0
2. APPLY edge-preserving bilateral denoising with a 5 x 5 neighbourhood
3. APPLY a mild unsharp mask (sigma 0.8) to existing ridge edges
4. BINARIZE each pixel against its 13 x 13 local intensity mean
5. THIN black ridge components to one-pixel centre lines
6. REMOVE false ridges shorter than 10 pixels and close small ridge gaps

SEPARATE ATTENDANCE FUNCTION (after enhancement)
COMPARE query to each canonical enrolment reference with SIFT-RANSAC geometry
FUSE secondary enhanced ORB, aligned structural and spectral evidence
IF similarity passes threshold and ambiguity guard
    RECORD attendance once for the selected class session
END IF""",
            language="text",
        )


def page_system() -> None:
    page_header(
        "Configuration and provenance",
        "System & help",
        "Review local storage, install a clearly labelled demo cohort and understand the prototype's privacy and decision boundaries.",
        "VERSION 2.0",
    )
    first, second = st.columns([1, 1.15], gap="large")
    with first:
        st.subheader("Quick start")
        st.markdown('<div class="mono-note">python -m streamlit run app.py</div>', unsafe_allow_html=True)
        st.caption(
            "Run from this project directory. In a new PowerShell window, activate "
            ".\\.venv\\Scripts\\Activate.ps1 first if python is not already on PATH."
        )

        st.subheader("Labelled demonstration data")
        st.write(
            "The demo cohort creates four synthetic fingerprint identities and one BMDS2133 session. "
            "Every demo name is visibly marked; these records must not be presented as experimental data."
        )
        if st.button("Load or repair demo cohort", type="primary", width="stretch"):
            try:
                students_added, references_upgraded, sessions_added = load_demo_cohort()
                st.success(
                    "Demo setup complete: "
                    f"{students_added} student(s) added, {references_upgraded} legacy reference(s) "
                    f"upgraded and {sessions_added} session(s) added."
                )
                st.session_state.pop("last_scan", None)
            except Exception as exc:
                st.error(f"Demo setup failed: {exc}")

        demo_scans = sorted(DEMO_SCAN_DIR.glob("*.png")) if DEMO_SCAN_DIR.exists() else []
        if demo_scans:
            st.caption(f"{len(demo_scans)} scanner samples are available on the Mark attendance page.")

        st.subheader("Local storage")
        st.code(
            f"Database: {DATABASE_PATH}\nReferences: {REFERENCE_DIR}\nEnhanced templates: {TEMPLATE_DIR}",
            language="text",
        )
        template_rows = list_templates()
        legacy_templates = sum(not row.get("reference_path") for row in template_rows)
        if legacy_templates:
            st.warning(
                f"{legacy_templates} existing template(s) pre-date canonical reference storage. "
                "They remain available through a limited enhanced-template fallback. Add new canonical "
                "captures on Student enrolment for reliable different-capture matching."
            )

    with second:
        st.subheader("System boundaries")
        st.markdown(
            """
            - **Local processing:** Images and biometric templates are not sent to a web API.
            - **Reference matching:** A consistently sized greyscale copy of each enrolment upload is retained as the canonical local reference alongside its enhanced binary result.
            - **Duplicate control:** A student can be recorded only once per class session.
            - **Transparent scores:** Reference SIFT, enhanced ORB, structural and spectral evidence remain visible for each decision.
            - **Browser camera capture:** A fresh camera still can trigger verification automatically, but it is not a certified liveness check.
            - **No liveness proof:** Uploads and camera stills remain vulnerable to replay without anti-spoofing or challenge-response controls.
            - **Prototype scope:** This system supports an image-processing assignment; it is not certified for forensic or high-security identity use.
            """
        )
        st.warning(
            "Fingerprint data is sensitive personal data. Obtain consent, restrict access to the data folder, "
            "define a retention period and use encrypted storage before any real institutional deployment."
        )

        st.subheader("Architecture")
        st.markdown(
            """
            **Input manager** validates image formats and supplies a consistent array size for database comparison.  
            **Enhancement engine** executes CLAHE contrast enhancement, bilateral edge-preserving denoising, mild unsharp enhancement, local-mean binarization, thinning and ridge repair.<br>
            **Matching service** compares each fresh capture to canonical enrolment references, then applies similarity, geometric-inlier and ambiguity guards.
            **Attendance database** integrates students, templates, sessions, records and audit events.  
            **Reporting module** exports complete class rosters to CSV and polished PDF.
            """
        )

        st.subheader("Assignment feature coverage")
        coverage = pd.DataFrame(
            [
                ["Preprocessing", "Implemented", "CLAHE corrects uneven local contrast with clipped amplification"],
                ["Main filter", "Implemented", "Bilateral filtering suppresses camera noise while preserving ridge boundaries"],
                ["Object detection", "Implemented", "Centre-seeded colour and boundary segmentation crops and normalizes the fingertip"],
                ["Data dashboard", "Implemented", "Quality, enhancement, matching and attendance summaries"],
                ["PDF reporting", "Implemented", "Automatic class attendance register with verification evidence"],
                ["Bulk GUI ingestion", "Implemented", "One to three same-finger captures through multi-image selection"],
                ["Supplemental functions", "Implemented", "Foreground guard, binary ridge repair, reference-score fusion and ambiguity guard"],
                ["Browser camera capture", "Implemented", "A captured camera still automatically starts enhancement, identification and attendance"],
                ["Continuous video liveness", "Future work", "Video tracking, challenge-response and anti-replay require a dedicated liveness design"],
            ],
            columns=["Rubric item", "Status", "System evidence"],
        )
        st.dataframe(coverage, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Recent audit activity")
    events = recent_audit_events(15)
    if events:
        frame = pd.DataFrame(events).rename(
            columns={"event_type": "Event", "details": "Details", "created_at": "Time"}
        )
        frame["Time"] = frame["Time"].map(fmt_datetime)
        st.dataframe(frame[["Time", "Event", "Details"]], width="stretch", hide_index=True)
    else:
        st.caption("Audit activity will appear after enrolment, session or attendance actions.")


PAGES = {
    "Dashboard": page_dashboard,
    "Mark attendance": page_mark_attendance,
    "Student enrolment": page_enrolment,
    "Class sessions": page_sessions,
    "Attendance records": page_records,
    "Algorithm studio": page_algorithm_studio,
    "System & help": page_system,
}


sidebar_brand()
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Dashboard"
st.sidebar.radio(
    "Navigation",
    options=list(PAGES),
    key="navigation",
    label_visibility="collapsed",
)
st.sidebar.markdown(
    """
    <div class="privacy-note">
        <b style="color:#9cb1a7">Local-only prototype</b><br/>
        Fingerprint enhancement, matching and attendance records remain on this computer.
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()
side_stats = dashboard_statistics()
st.sidebar.caption(
    f"{side_stats['students']} students / {side_stats['active_sessions']} live sessions / ridge-preserving enhancement"
)

PAGES[st.session_state["navigation"]]()
