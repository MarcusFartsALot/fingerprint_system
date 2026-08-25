"""Shared visual language for the Streamlit interface."""

from __future__ import annotations

from html import escape

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #09110e;
            --panel: #101b17;
            --panel-2: #14231d;
            --line: rgba(211, 238, 225, 0.11);
            --text: #edf7f2;
            --muted: #91a79d;
            --green: #36d994;
            --green-deep: #0c8b5c;
            --amber: #f0b44d;
            --red: #f16f6f;
        }

        html, body, [class*="css"] {
            font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 90% -10%, rgba(35, 148, 101, 0.17), transparent 32rem),
                radial-gradient(circle at 20% 110%, rgba(23, 96, 70, 0.12), transparent 35rem),
                var(--bg);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #0c1512;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
        .block-container {
            max-width: 1480px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 { letter-spacing: -0.035em; }
        h1 { font-size: 2.35rem !important; }
        h2 { font-size: 1.48rem !important; margin-top: 1.4rem !important; }
        h3 { font-size: 1.03rem !important; }
        p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
        hr { border-color: var(--line) !important; }

        .brand {
            display: flex;
            gap: .75rem;
            align-items: center;
            padding: .25rem .15rem 1.15rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }
        .brand-mark {
            width: 2.4rem;
            height: 2.4rem;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, #3be39b, #118257);
            color: #06110c;
            font-weight: 900;
            font-size: 1.05rem;
            box-shadow: 0 10px 30px rgba(54, 217, 148, .18);
        }
        .brand-name { color: var(--text); font-size: .96rem; font-weight: 760; line-height: 1.05; }
        .brand-sub { color: var(--muted); font-size: .69rem; margin-top: .22rem; }

        .page-head {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            align-items: flex-end;
            padding-bottom: 1.4rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.35rem;
        }
        .page-kicker {
            color: var(--green);
            font-size: .73rem;
            letter-spacing: .14em;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: .48rem;
        }
        .page-title { color: var(--text); font-size: 2.2rem; font-weight: 780; letter-spacing: -.045em; line-height: 1.02; }
        .page-description { color: var(--muted); max-width: 48rem; font-size: .92rem; line-height: 1.55; margin-top: .62rem; }
        .head-badge {
            color: #aef1d2;
            background: rgba(54, 217, 148, .1);
            border: 1px solid rgba(54, 217, 148, .24);
            padding: .46rem .7rem;
            border-radius: 999px;
            white-space: nowrap;
            font-size: .72rem;
            font-weight: 700;
        }

        .stat-card {
            background: linear-gradient(145deg, rgba(19, 34, 28, .98), rgba(13, 25, 20, .98));
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 1rem 1.05rem;
            min-height: 7.1rem;
            position: relative;
            overflow: hidden;
        }
        .stat-card::after {
            content: "";
            width: 5rem;
            height: 5rem;
            border-radius: 50%;
            background: rgba(54, 217, 148, .055);
            position: absolute;
            top: -2.4rem;
            right: -1.8rem;
        }
        .stat-label { color: var(--muted); font-size: .73rem; font-weight: 650; }
        .stat-value { color: var(--text); font-size: 1.72rem; font-weight: 790; letter-spacing: -.04em; margin-top: .62rem; line-height: 1; }
        .stat-note { color: #698178; font-size: .68rem; margin-top: .52rem; }

        .section-card, div[data-testid="stForm"] {
            background: rgba(16, 27, 23, .92);
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 1rem 1.05rem;
        }
        .session-strip {
            display: grid;
            grid-template-columns: minmax(12rem, 1.4fr) 1fr 1fr auto;
            gap: 1rem;
            align-items: center;
            background: linear-gradient(105deg, rgba(19, 50, 38, .98), rgba(14, 28, 22, .98));
            border: 1px solid rgba(54, 217, 148, .18);
            border-radius: 17px;
            padding: 1rem 1.1rem;
            margin: .45rem 0 .9rem;
        }
        .strip-label { color: #789086; font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }
        .strip-value { color: var(--text); font-size: .88rem; font-weight: 680; margin-top: .24rem; }
        .live-dot { width: .47rem; height: .47rem; background: var(--green); border-radius: 50%; display: inline-block; margin-right: .35rem; box-shadow: 0 0 0 5px rgba(54,217,148,.09); }

        .result-success, .result-fail, .result-warning {
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            margin: .6rem 0 1rem;
        }
        .result-success { background: rgba(23, 151, 101, .12); border: 1px solid rgba(54, 217, 148, .32); }
        .result-fail { background: rgba(216, 75, 75, .10); border: 1px solid rgba(241, 111, 111, .28); }
        .result-warning { background: rgba(222, 154, 45, .10); border: 1px solid rgba(240, 180, 77, .28); }
        .result-title { color: var(--text); font-size: 1.15rem; font-weight: 770; }
        .result-copy { color: var(--muted); font-size: .83rem; margin-top: .34rem; line-height: 1.45; }

        .pipeline {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .55rem;
            min-width: 0;
            margin: .8rem 0 1.2rem;
        }
        .pipe-step { background: var(--panel); border: 1px solid var(--line); padding: .76rem; border-radius: 13px; }
        .pipe-num { color: var(--green); font-size: .66rem; font-weight: 800; letter-spacing: .08em; }
        .pipe-name { color: var(--text); font-size: .76rem; font-weight: 680; margin-top: .32rem; }
        .pipe-copy { color: #72877e; font-size: .63rem; line-height: 1.35; margin-top: .25rem; }

        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
        [data-testid="stFileUploader"] {
            background: rgba(16, 27, 23, .7);
            border-radius: 15px;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(11, 22, 17, .7);
            border-color: rgba(54, 217, 148, .18);
        }
        .stButton > button, .stDownloadButton > button, button[kind="primary"] {
            border-radius: 11px !important;
            min-height: 2.55rem;
            font-weight: 680 !important;
            border-color: rgba(54, 217, 148, .25) !important;
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #23c982, #0e8d5d) !important;
            color: #04100b !important;
        }
        [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input {
            background-color: #101c17 !important;
            border-color: var(--line) !important;
            border-radius: 10px !important;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: .8rem .9rem;
        }
        [data-testid="stMetricValue"] { color: var(--text); }
        [data-testid="stAlert"] { border-radius: 13px; }
        [data-testid="stImage"] img { border-radius: 13px; border: 1px solid var(--line); }
        .mono-note {
            border: 1px solid var(--line);
            background: #070d0a;
            color: #b8d4c8;
            font-family: "Cascadia Code", Consolas, monospace;
            padding: .75rem .85rem;
            border-radius: 11px;
            font-size: .78rem;
        }
        .privacy-note { color: #6f857b; font-size: .67rem; line-height: 1.45; padding: .75rem .15rem 0; }

        @media (max-width: 900px) {
            .pipeline { grid-template-columns: repeat(2, 1fr); }
            .session-strip { grid-template-columns: 1fr 1fr; }
            .page-head { align-items: flex-start; flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="brand">
            <div class="brand-mark">ST</div>
            <div>
                <div class="brand-name">Fingerprint Attendance</div>
                <div class="brand-sub">Fingerprint verification console</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, description: str, badge: str = "EIGHT STAGES") -> None:
    st.markdown(
        f"""
        <div class="page-head">
            <div>
                <div class="page-kicker">{escape(kicker)}</div>
                <div class="page-title">{escape(title)}</div>
                <div class="page-description">{escape(description)}</div>
            </div>
            <div class="head-badge"><span class="live-dot"></span>{escape(badge)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{escape(label)}</div>
            <div class="stat-value">{escape(value)}</div>
            <div class="stat-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_banner(kind: str, title: str, copy: str) -> None:
    css_class = {"success": "result-success", "warning": "result-warning"}.get(kind, "result-fail")
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="result-title">{escape(title)}</div>
            <div class="result-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_strip() -> None:
    steps = [
        ("01", "Local histogram equalization", "Expand ridge contrast in each 11 x 11 neighbourhood"),
        ("02", "Adaptive Wiener filtering", "Suppress local noise with a pixel-wise 3 x 3 estimate"),
        ("03", "Local-mean binarization", "Compare each pixel with its 13 x 13 neighbourhood mean"),
        ("04", "Morphological thinning", "Reduce black ridges to one-pixel-wide centre lines"),
        ("05", "Binary ridge post-processing", "Remove short false ridges and close small gaps"),
    ]
    cards = "".join(
        f'<div class="pipe-step"><div class="pipe-num">{number}</div><div class="pipe-name">{name}</div><div class="pipe-copy">{copy}</div></div>'
        for number, name, copy in steps
    )
    st.markdown(f'<div class="pipeline">{cards}</div>', unsafe_allow_html=True)
