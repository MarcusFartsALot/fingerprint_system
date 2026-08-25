"""Paper-aligned student attendance fingerprint system.

Run from this directory with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import importlib
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import fingerprint_processing as fingerprint_module

from database import (
    DATA_DIR,
    DATABASE_PATH,
    REFERENCE_DIR,
    TEMPLATE_DIR,
    add_fingerprint_templates,
    attendance_records,
    close_session,
    create_session,
    dashboard_statistics,
    enrol_student,
    get_session,
    get_student,
    initialise_database,
    list_sessions,
    list_students,
    list_templates,
    local_now,
    mark_attendance,
    recent_audit_events,
    session_roster,
)
# Streamlit can retain an older imported module while hot-reloading this file.
# Reload only when the processing module's public interface has changed.
_REQUIRED_PIPELINE_SCHEMA_VERSION = "greenberg-filtering-v4"
_PROCESSING_EXPORTS = (
    "PIPELINE_SCHEMA_VERSION",
    "PipelineResult",
    "generate_demo_fingerprint",
    "process_fingerprint",
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
from matching import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MATCH_THRESHOLD,
    IdentificationResult,
    attendance_status,
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
    return f"{session['course_code']} - {fmt_datetime(session['starts_at'], short=True)} - {live}"


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
                "minutiae_count": 0,
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
        st.metric("Match confidence", f"{similarity:.1%}")
    with metrics[1]:
        st.metric("Capture quality", f"{result.quality['overall']:.1%}", quality_label(result.quality["overall"]))
    with metrics[2]:
        st.metric("Ridge continuity", f"{result.quality['connectivity']:.1%}", "After post-processing")
    with metrics[3]:
        st.metric("Processing", f"{payload['elapsed_ms']:.0f} ms", f"{identification.candidates_checked} templates")

    image_columns = st.columns(3)
    with image_columns[0]:
        st.image(result.prepared, caption="Original input", width="stretch", clamp=True)
    with image_columns[1]:
        st.image(result.wiener_filtered, caption="2. Adaptive Wiener filtering", width="stretch", clamp=True)
    with image_columns[2]:
        st.image(result.enhanced, caption="5. Enhanced binary fingerprint", width="stretch", clamp=True)

    with st.expander("Technical decision evidence"):
        st.caption(f"Executed enhancement: {' -> '.join(result.stages)}.")
        if identification.evidence:
            evidence = identification.evidence
            evidence_columns = st.columns(5)
            evidence_columns[0].metric(
                "Reference geometry",
                f"{evidence.reference_score:.1%}",
                f"{evidence.reference_inliers}/{evidence.reference_matches} SIFT inliers",
            )
            evidence_columns[1].metric(
                "ORB geometry",
                f"{evidence.orb_score:.1%}",
                f"{evidence.geometric_inliers}/{evidence.good_matches} inlier pairs",
            )
            evidence_columns[2].metric("Structural", f"{evidence.structural_score:.1%}", "Aligned ridge flow")
            evidence_columns[3].metric("Spectral", f"{evidence.spectral_score:.1%}", "Local ridge spectrum")
            evidence_columns[4].metric("Runner-up", f"{identification.runner_up_similarity:.1%}", "Ambiguity guard")
            reference_mode = "canonical enrolment upload" if evidence.used_canonical_reference else "legacy enhanced-template fallback"
            st.caption(f"Primary comparison source: {reference_mode}.")
        diagnostic_columns = st.columns(5)
        with diagnostic_columns[0]:
            st.image(result.local_equalised, caption="1. Local histogram equalization", width="stretch")
        with diagnostic_columns[1]:
            st.image(result.wiener_filtered, caption="2. Adaptive Wiener filtering", width="stretch")
        with diagnostic_columns[2]:
            st.image(result.binary, caption="3. Adaptive local-mean binary", width="stretch")
        with diagnostic_columns[3]:
            st.image(result.thinned, caption="4. Morphological thinning", width="stretch")
        with diagnostic_columns[4]:
            st.image(result.enhanced, caption="5. Binary ridge post-processing", width="stretch")
        st.image(
            result.region_mask,
            caption="System extension: variance foreground guard (not a numbered paper stage)",
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
        "Select a live class, provide one fingerprint capture and let the paper-aligned filtering pipeline identify the enrolled student.",
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

    source_mode = st.radio("Fingerprint source", ["Upload scanner image", "Use labelled demo scan"], horizontal=True)
    source = None
    if source_mode == "Upload scanner image":
        uploaded = st.file_uploader(
            "Drop a fingerprint capture",
            type=IMAGE_TYPES,
            accept_multiple_files=False,
            help="PNG, JPG, BMP or TIFF. Greyscale conversion is automatic.",
        )
        if uploaded:
            source = uploaded.getvalue()
    else:
        demo_scans = sorted(DEMO_SCAN_DIR.glob("*.png")) if DEMO_SCAN_DIR.exists() else []
        if not demo_scans:
            st.info("Load the labelled demo cohort from System & help to enable demo scanner inputs.")
        else:
            demo_map = {path.stem.replace("_", " "): path for path in demo_scans}
            chosen_demo = st.selectbox("Demo scanner capture", list(demo_map))
            source = demo_map[chosen_demo]

    if st.button("Enhance, verify and mark attendance", type="primary", width="stretch"):
        if source is None:
            st.warning("Provide a fingerprint capture first.")
        elif not list_templates():
            st.warning("No enrolled fingerprint templates are available.")
        else:
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
                    with st.spinner("Running the paper-aligned local filtering stages..."):
                        for upload in uploads:
                            result = process_fingerprint(upload.getvalue())
                            if result.quality["overall"] < 0.20:
                                raise ValueError(
                                    f"{upload.name} has insufficient ridge quality ({result.quality['overall']:.0%})."
                                )
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
                                    "minutiae_count": 0,
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
            preview_columns = st.columns(3)
            preview_columns[0].image(preview.prepared, caption="Original input", width="stretch")
            preview_columns[1].image(preview.wiener_filtered, caption="2. Wiener-filtered image", width="stretch")
            preview_columns[2].image(preview.enhanced, caption="5. Stored enhanced binary", width="stretch")
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
                                if result.quality["overall"] < 0.20:
                                    raise ValueError(
                                        f"{upload.name} has insufficient ridge quality "
                                        f"({result.quality['overall']:.0%})."
                                    )
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
                                        "minutiae_count": 0,
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
            if st.button(
                f"Close {session['course_code']} session",
                key=f"close_{session['session_id']}",
            ):
                close_session(session["session_id"])
                st.success("Session closed. Its attendance register remains available.")
                st.rerun()

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

    session_options = {"All verified records": None}
    session_options.update({session_label(session): session["session_id"] for session in sessions})
    filters = st.columns([1.4, 1])
    chosen = filters[0].selectbox("Class session", list(session_options))
    search = filters[1].text_input("Search student or programme", placeholder="Name, ID or programme")
    selected_id = session_options[chosen]

    if selected_id is None:
        records = attendance_records(search=search)
        frame = results_dataframe(records)
        total = len(records)
        present = sum(record["attendance_status"] == "Present" for record in records)
        late = sum(record["attendance_status"] == "Late" for record in records)
        stats = st.columns(3)
        with stats[0]:
            stat_card("Verified records", str(total), "Across every class session")
        with stats[1]:
            stat_card("Present", str(present), "Within the grace period")
        with stats[2]:
            stat_card("Late", str(late), "After the grace period")
        if not frame.empty:
            st.dataframe(frame, width="stretch", hide_index=True)
            st.download_button(
                "Download verified records (CSV)",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name="attendance_records.csv",
                mime="text/csv",
            )
        else:
            st.caption("No verified attendance matches the current filter.")
    else:
        session = get_session(selected_id)
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
        stats = st.columns(4)
        with stats[0]:
            stat_card("Roster", str(len(roster)), f"{session['course_code']} enrolled identities")
        with stats[1]:
            stat_card("Present", str(present), "Within the grace period")
        with stats[2]:
            stat_card("Late", str(late), "Verified after the grace period")
        with stats[3]:
            rate = (present + late) / len(roster) if roster else 0
            stat_card("Attendance rate", f"{rate:.0%}", f"{absent} absent")

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
            display["Marked at"] = display["Marked at"].map(fmt_datetime)
            display["Match"] = display["Match"].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "-")
            display["Quality"] = display["Quality"].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "-")
            st.dataframe(
                display[["Student ID", "Student", "Programme", "Year", "Group", "Status", "Marked at", "Match", "Quality"]],
                width="stretch",
                hide_index=True,
            )
            csv_data = display.to_csv(index=False).encode("utf-8")
            export_columns = st.columns(2)
            export_columns[0].download_button(
                "Download class register (CSV)",
                data=csv_data,
                file_name=f"{session['course_code']}_attendance.csv",
                mime="text/csv",
                width="stretch",
            )
            pdf_bytes = build_attendance_pdf(session, roster)
            export_columns[1].download_button(
                "Download polished attendance report (PDF)",
                data=pdf_bytes,
                file_name=f"{session['course_code']}_attendance_report.pdf",
                mime="application/pdf",
                width="stretch",
            )


def render_studio_result(result: PipelineResult) -> None:
    required_outputs = (
        "stages",
        "region_mask",
        "local_equalised",
        "wiener_filtered",
        "binary",
        "thinned",
        "enhanced",
    )
    if any(not hasattr(result, output) for output in required_outputs):
        st.session_state.pop("studio_result", None)
        st.warning(
            "The cached diagnostic result used an older processing pipeline. "
            "Upload the image and run the paper-aligned filtering pipeline again."
        )
        return
    st.info(f"Fixed executed sequence: **{' -> '.join(result.stages)}**.")
    metrics = st.columns(5)
    metrics[0].metric("Overall quality", f"{result.quality['overall']:.1%}")
    metrics[1].metric("Local contrast", f"{result.quality['contrast']:.1%}")
    metrics[2].metric("Noise suppression", f"{result.quality['noise_reduction']:.1%}")
    metrics[3].metric("Ridge continuity", f"{result.quality['connectivity']:.1%}")
    metrics[4].metric("Runtime", f"{result.processing_ms:.0f} ms")

    stage_columns = st.columns(5)
    stage_columns[0].image(result.local_equalised, caption="1. Local histogram equalization", width="stretch")
    stage_columns[1].image(result.wiener_filtered, caption="2. Adaptive Wiener filtering", width="stretch")
    stage_columns[2].image(result.binary, caption="3. Local-mean binarization", width="stretch")
    stage_columns[3].image(result.thinned, caption="4. Morphological thinning", width="stretch")
    stage_columns[4].image(result.enhanced, caption="5. Binary ridge post-processing", width="stretch")
    with st.expander("System foreground extension"):
        st.image(result.region_mask, caption="Variance-based foreground guard", width=320)
        st.caption(
            "This mask excludes scanner borders and blank background before the paper stages. "
            "It is an implementation extension, not presented as one of Greenberg et al.'s numbered filters."
        )


def page_algorithm_studio() -> None:
    page_header(
        "Image processing evidence",
        "Fingerprint enhancement studio",
        "Inspect each filtering result from local contrast enhancement to the final cleaned binary ridges.",
        "FIVE PAPER STAGES",
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
                with st.spinner("Running the complete Greenberg filtering pipeline..."):
                    st.session_state["studio_result"] = process_fingerprint(upload.getvalue())
            except Exception as exc:
                st.error(f"The image could not be analysed: {exc}")
    if "studio_result" in st.session_state:
        render_studio_result(st.session_state["studio_result"])

    st.divider()
    st.subheader("Paper method comparison")
    st.caption(
        "Method properties taken from the selected paper. The application does not invent accuracy values; "
        "submit quantitative claims only after running a labelled evaluation dataset."
    )
    comparison = pd.DataFrame(
        [
            ["Binary filtering (implemented)", "Local histogram + Wiener", "No", "Binary skeleton", "Production pipeline"],
            ["Modified Gabor", "Orientation + ridge frequency", "Yes", "Greyscale", "Paper comparison method"],
            ["Anisotropic filtering", "Local orientation", "No", "Greyscale", "Paper comparison method"],
        ],
        columns=["Method", "Main information", "Frequency required", "Output", "Use here"],
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.info(
        "The previous Assignment IP table is historical report evidence and was not rerun by this app. "
        "The live pipeline now implements the filtering technique in the selected Greenberg paper."
    )

    with st.expander("Implementation pseudocode"):
        st.code(
            """INPUT greyscale fingerprint capture
SYSTEM EXTENSION: locate fingerprint foreground and reject scanner borders
1. APPLY local histogram equalization with an 11 x 11 neighbourhood
2. APPLY pixel-wise adaptive Wiener filtering with a 3 x 3 neighbourhood
3. BINARIZE each pixel against its 13 x 13 local intensity mean
4. THIN black ridge components to one-pixel centre lines
5. REMOVE false ridges shorter than 10 pixels and close small ridge gaps

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
            **Enhancement engine** executes local histogram equalization, adaptive Wiener filtering, adaptive local-mean binarization, thinning and binary ridge repair.<br>
            **Matching service** compares each scan to canonical enrolment uploads, then fuses enhanced and structural evidence with threshold and ambiguity guards.  
            **Attendance database** integrates students, templates, sessions, records and audit events.  
            **Reporting module** exports complete class rosters to CSV and polished PDF.
            """
        )

        st.subheader("Assignment feature coverage")
        coverage = pd.DataFrame(
            [
                ["Preprocessing", "Implemented", "Local histogram equalization, adaptive Wiener noise reduction and local-mean binarization"],
                ["Image calibration", "Not included", "Removed to follow the requested enhancement diagram exactly"],
                ["Object detection", "Implemented", "Connected fingerprint segmentation rejects scanner frames and empty background"],
                ["Data dashboard", "Implemented", "Quality, enhancement, matching and attendance summaries"],
                ["PDF reporting", "Implemented", "Automatic class attendance register with verification evidence"],
                ["Bulk GUI ingestion", "Implemented", "One to three same-finger captures through multi-image selection"],
                ["Supplemental functions", "Implemented", "Foreground guard, binary ridge repair, reference-score fusion and ambiguity guard"],
                ["Video processing", "Not applicable", "Attendance verification uses deliberate still scanner captures"],
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
    f"{side_stats['students']} students / {side_stats['active_sessions']} live sessions / paper-aligned filtering"
)

PAGES[st.session_state["navigation"]]()
