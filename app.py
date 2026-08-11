from __future__ import annotations

import io
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import streamlit as st

from thermal_lab_core import (
    COLD_INTERFACE_M,
    CONDUCTION_COLUMNS,
    HOT_INTERFACE_M,
    RADIATION_COLUMNS,
    THERMOCOUPLE_POSITIONS_M,
    analyse_conduction,
    analyse_radiation,
    blank_conduction_data,
    blank_radiation_data,
    conduction_uncertainty_percent,
    demonstration_conduction_data,
    demonstration_radiation_data,
    equilibrium_sensor_temperature_C,
    forced_convection_h,
    linearised_radiation_coefficient_W_m2K,
    radiation_corrected_medium_temperature_C,
    valid_conduction_rows,
    valid_radiation_rows,
)


st.set_page_config(
    page_title="ThermalLab | ME3512",
    page_icon="♨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
APP_VERSION = "1.0"
JCU_ID_PATTERN = re.compile(r"^\d{8}$")

PRACTICAL_1 = "Practical 1 - Linear conduction"
PRACTICAL_2 = "Practical 2 - Radiation measurement error"
PRACTICALS = [PRACTICAL_1, PRACTICAL_2]

SECTIONS = [
    "Prepare",
    "Predict",
    "Apparatus and procedure",
    "Record data",
    "Calculate and visualise",
    "Explore the concept",
    "Interpret results",
    "Review and download",
]

VIDEO_URLS = {
    PRACTICAL_1: "https://www.youtube.com/watch?v=fwX8ic8N6ko",
    PRACTICAL_2: "https://www.youtube.com/watch?v=a24syZRWj3k",
}

PRACTICAL_TITLES = {
    PRACTICAL_1: "Contact resistance in linear heat conduction",
    PRACTICAL_2: "Radiation error in thermocouple measurements",
}

PRACTICAL_SUBTITLES = {
    PRACTICAL_1: "See the interface temperature jumps that a simple Fourier-law calculation can hide.",
    PRACTICAL_2: "Discover why a thermocouple can be stable, repeatable and still report the wrong air temperature.",
}


def initialise_state() -> None:
    defaults = {
        "student_name": "",
        "student_id": "",
        "group": "",
        "lab_date": date.today(),
        "pathway": "Physical laboratory",
        "practical": PRACTICAL_1,
        "conduction_data": blank_conduction_data(),
        "radiation_data": blank_radiation_data(),
        "cond_editor_version": 0,
        "rad_editor_version": 0,
        "cond_quiz_submitted": False,
        "rad_quiz_submitted": False,
        "cond_video_ready": False,
        "rad_video_ready": False,
        "cond_safety": {},
        "rad_safety": {},
        "cond_explore_complete": False,
        "rad_explore_complete": False,
        "cond_interpret_1": "",
        "cond_interpret_2": "",
        "cond_interpret_3": "",
        "cond_interpret_4": "",
        "rad_interpret_1": "",
        "rad_interpret_2": "",
        "rad_interpret_3": "",
        "rad_interpret_4": "",
        "diameter_mm": 25.0,
        "heat_fraction": 1.0,
        "brass_reference_k": 119.0,
        "aluminium_reference_k": 180.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_state()


st.markdown(
    """
    <style>
      :root {
        --jcu-blue:#0B4F8A; --jcu-blue-dark:#073B68; --ink:#172033;
        --muted:#64748B; --line:#DCE4EC; --soft:#F5F8FB; --teal:#0F766E;
        --orange:#C25A10; --red:#B42318; --green:#166534;
      }
      html { scroll-behavior:smooth; }
      .stApp { background:#FFFFFF; color:var(--ink); }
      .block-container { max-width:1420px; padding-top:0.8rem; padding-bottom:3rem; }
      header[data-testid="stHeader"] { background:transparent; }
      #MainMenu, footer { visibility:hidden; }
      [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#F8FAFC 0%,#F1F5F9 65%,#EAF0F5 100%);
        border-right:1px solid #D7E0E9; min-width:285px; max-width:305px;
      }
      [data-testid="stSidebar"] > div:first-child { padding-top:0.65rem; }
      [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:0.28rem; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background:rgba(255,255,255,0.90); border:1px solid #DBE3EC;
        border-radius:10px; padding:0.50rem 0.62rem; margin:0.05rem 0;
        transition:all 0.16s ease;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        border-color:#A8BED1; background:#FFFFFF;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:#EDF5FC; border-color:#88ACCA;
        box-shadow:inset 3px 0 0 var(--jcu-blue),0 1px 3px rgba(15,23,42,.05);
      }
      [data-testid="stSidebar"] [data-testid="stProgress"] > div {
        height:10px; border-radius:999px; background:#D9E4ED;
      }
      [data-testid="stSidebar"] [data-testid="stProgress"] > div > div {
        border-radius:999px; background:linear-gradient(90deg,#0B4F8A,#1482B8);
        box-shadow:0 0 8px rgba(11,79,138,.20);
      }
      .sidebar-brand { padding:6px 5px 12px 13px; margin:1px 0 11px; border-left:3px solid var(--jcu-blue); }
      .sidebar-brand .course { color:#617386; font-size:9.5px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
      .sidebar-brand .title { color:#153A59; font-size:22px; font-weight:850; line-height:1.08; margin-top:4px; }
      .sidebar-brand .subtitle { color:#68788A; font-size:11px; line-height:1.42; margin-top:6px; max-width:240px; }
      .sidebar-label { color:#526274; font-size:10px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin:14px 2px 7px; }
      .sidebar-card { background:rgba(255,255,255,.78); border:1px solid #DCE4EC; border-radius:10px; padding:10px 11px; margin:8px 0; }
      .sidebar-card .label { color:#64748B; font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
      .sidebar-card .value { color:#172033; font-size:13px; font-weight:750; margin-top:3px; }
      .sidebar-card .meta { color:#64748B; font-size:11px; line-height:1.4; margin-top:4px; }
      .progress-head { display:flex; justify-content:space-between; color:#334155; font-size:12px; font-weight:800; margin:5px 1px; }
      .progress-head span { color:var(--jcu-blue); }
      .progress-meta { color:#738194; font-size:11px; margin:5px 1px 12px; }
      .sidebar-footer { color:#7A8796; font-size:10.5px; line-height:1.45; margin-top:14px; padding:10px 3px 2px; border-top:1px solid #D7DFE8; }
      .app-header { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:25px; align-items:center; background:linear-gradient(112deg,#073B68 0%,#0B4F8A 58%,#126B98 100%); border-radius:18px; padding:22px 26px; color:white; box-shadow:0 8px 26px rgba(7,59,104,.17); margin-bottom:17px; }
      .app-header .eyebrow { font-size:11px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; opacity:.83; }
      .app-header h1 { font-size:30px; line-height:1.10; margin:5px 0 5px; color:white; }
      .app-header .subtitle { font-size:13.5px; line-height:1.48; color:#E7F2FA; max-width:850px; }
      .header-mark { min-width:170px; text-align:right; }
      .header-mark .jcu { font-size:18px; font-weight:850; letter-spacing:.02em; }
      .header-mark .school { font-size:10px; opacity:.8; margin-top:4px; }
      .section-head { display:flex; gap:15px; align-items:flex-start; margin:17px 0 13px; }
      .section-number { width:42px; height:42px; flex:0 0 42px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:#E9F3FB; border:1px solid #C8DDED; color:#0B4F8A; font-weight:850; font-size:17px; }
      .section-head h2 { margin:0; color:#183A57; font-size:25px; line-height:1.15; }
      .section-head p { margin:5px 0 0; color:#64748B; font-size:13px; line-height:1.45; }
      .hero-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:10px 0 16px; }
      .info-card { background:#FFFFFF; border:1px solid #DBE4EC; border-radius:13px; padding:15px 16px; box-shadow:0 2px 8px rgba(15,23,42,.035); }
      .info-card .kicker { font-size:10px; text-transform:uppercase; letter-spacing:.1em; font-weight:850; color:#60758A; }
      .info-card .title { font-size:15px; font-weight:800; color:#193B58; margin:5px 0 5px; }
      .info-card .text { color:#5F6F80; font-size:12.5px; line-height:1.48; }
      .concept-card { border:1px solid #BBD7E8; background:linear-gradient(135deg,#F2F9FD,#FFFFFF); border-radius:14px; padding:17px 18px; margin:12px 0 16px; }
      .concept-card .label { color:#0B4F8A; font-size:10px; font-weight:850; letter-spacing:.11em; text-transform:uppercase; }
      .concept-card .big { color:#123A5A; font-size:20px; font-weight:850; margin:5px 0; }
      .concept-card .text { color:#536779; font-size:13px; line-height:1.55; }
      .status-strip { border-radius:10px; padding:10px 12px; font-size:12.5px; line-height:1.45; margin:8px 0; }
      .status-strip.success { color:#166534; background:#ECFDF3; border:1px solid #BBF7D0; }
      .status-strip.warning { color:#854D0E; background:#FFFBEB; border:1px solid #FDE68A; }
      .status-strip.info { color:#164E63; background:#ECFEFF; border:1px solid #A5F3FC; }
      .metric-row { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:8px 0 15px; }
      .metric-card { background:#F8FAFC; border:1px solid #DDE5ED; border-radius:12px; padding:12px 14px; }
      .metric-card .label { color:#64748B; font-size:10.5px; font-weight:750; }
      .metric-card .value { color:#153A59; font-size:21px; font-weight:850; margin-top:4px; }
      .metric-card .meta { color:#758495; font-size:10.5px; margin-top:3px; }
      .equation-box { background:#F8FAFC; border-left:4px solid #0B4F8A; border-radius:8px; padding:12px 15px; margin:10px 0; color:#243A4D; }
      .evidence { background:#F7FBF8; border:1px solid #CFE6D3; border-radius:12px; padding:13px 15px; margin:10px 0; }
      .evidence b { color:#166534; }
      .small-note { color:#6B7C8E; font-size:11.5px; line-height:1.45; }
      div[data-testid="stDataEditor"] { border:1px solid #D9E2EA; border-radius:11px; overflow:hidden; }
      .stButton > button, .stDownloadButton > button { border-radius:9px; font-weight:750; }
      @media (max-width:900px) {
        .hero-grid,.metric-row { grid-template-columns:1fr 1fr; }
        .app-header { grid-template-columns:1fr; }
        .header-mark { text-align:left; }
      }
      @media (max-width:620px) { .hero-grid,.metric-row { grid-template-columns:1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_escape(value: object) -> str:
    import html

    return html.escape(str(value))


def practical_prefix(practical: str) -> str:
    return "cond" if practical == PRACTICAL_1 else "rad"


def identity_complete() -> bool:
    return bool(st.session_state.student_name.strip()) and bool(
        JCU_ID_PATTERN.fullmatch(st.session_state.student_id.strip())
    )


def section_heading(section: str, description: str) -> None:
    index = SECTIONS.index(section) + 1
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-number">{index}</div>
          <div><h2>{html_escape(section)}</h2><p>{html_escape(description)}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(practical: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="eyebrow">ME3512 · Heat and Mass Transfer · ThermalLab</div>
            <h1>{html_escape(PRACTICAL_TITLES[practical])}</h1>
            <div class="subtitle">{html_escape(PRACTICAL_SUBTITLES[practical])}</div>
          </div>
          <div class="header-mark">
            <div class="jcu">JAMES COOK UNIVERSITY</div>
            <div class="school">Engineering · practical learning companion</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="metric-card"><div class="label">{html_escape(label)}</div>'
        f'<div class="value">{html_escape(value)}</div><div class="meta">{html_escape(meta)}</div></div>'
        for label, value, meta in items
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def info_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="info-card"><div class="kicker">{html_escape(kicker)}</div>'
        f'<div class="title">{html_escape(title)}</div><div class="text">{html_escape(text)}</div></div>'
        for kicker, title, text in items
    )
    st.markdown(f'<div class="hero-grid">{cards}</div>', unsafe_allow_html=True)


def completion_checks(practical: str) -> list[bool]:
    prefix = practical_prefix(practical)
    if practical == PRACTICAL_1:
        valid_data = len(valid_conduction_rows(st.session_state.conduction_data)) >= 4
        safety = all(st.session_state.cond_safety.values()) and len(st.session_state.cond_safety) == 5
        interpretations = all(st.session_state.get(f"cond_interpret_{i}", "").strip() for i in range(1, 5))
    else:
        valid_data = len(valid_radiation_rows(st.session_state.radiation_data)) >= 4
        safety = all(st.session_state.rad_safety.values()) and len(st.session_state.rad_safety) == 5
        interpretations = all(st.session_state.get(f"rad_interpret_{i}", "").strip() for i in range(1, 5))
    return [
        identity_complete() and bool(st.session_state.get(f"{prefix}_video_ready")),
        bool(st.session_state.get(f"{prefix}_quiz_submitted")),
        safety,
        valid_data,
        valid_data,
        bool(st.session_state.get(f"{prefix}_explore_complete")),
        interpretations,
        False,
    ]


def render_sidebar() -> tuple[str, str]:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="course">ME3512 · Heat and Mass Transfer</div>
          <div class="title">ThermalLab</div>
          <div class="subtitle">Measure carefully. Model honestly. Explain physically.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-label">Choose experiment</div>', unsafe_allow_html=True)
    practical = st.sidebar.selectbox(
        "Practical",
        PRACTICALS,
        key="practical",
        label_visibility="collapsed",
    )
    checks = completion_checks(practical)
    completed = sum(checks[:7])
    progress = completed / 7.0
    st.sidebar.markdown(
        f'<div class="progress-head"><div>Progress</div><span>{round(progress*100)}%</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.progress(progress)
    st.sidebar.markdown(
        f'<div class="progress-meta">{completed} of 7 learning stages complete</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-label">Practical workflow</div>', unsafe_allow_html=True)
    section = st.sidebar.radio(
        "Workflow",
        SECTIONS,
        key=f"section_{practical_prefix(practical)}",
        label_visibility="collapsed",
    )
    name = st.session_state.student_name.strip() or "Student details not entered"
    student_id = st.session_state.student_id.strip() or "JCU ID pending"
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
          <div class="label">Current session</div>
          <div class="value">{html_escape(name)}</div>
          <div class="meta">{html_escape(student_id)} · {html_escape(st.session_state.pathway)}</div>
        </div>
        <div class="sidebar-footer">Entries stay in this browser session. Download your files before closing the app.<br><br>ThermalLab v{APP_VERSION} · Dr Mehdi Khatamifar</div>
        """,
        unsafe_allow_html=True,
    )
    return practical, section


def conduction_schematic() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(-0.013, 0.118)
    ax.set_ylim(-0.65, 1.10)
    ax.axis("off")
    segments = [
        (-0.004, HOT_INTERFACE_M, "Heated brass section", "#F5A45D"),
        (HOT_INTERFACE_M, COLD_INTERFACE_M, "Interchangeable sample", "#E7C86E"),
        (COLD_INTERFACE_M, 0.112, "Cooled brass section", "#76B8D8"),
    ]
    for left, right, label, colour in segments:
        ax.add_patch(Rectangle((left, 0.05), right - left, 0.47, facecolor=colour, edgecolor="#334155", linewidth=1.3))
        ax.text((left + right) / 2, 0.75, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#23384B")
    for index, x in enumerate(THERMOCOUPLE_POSITIONS_M, start=1):
        ax.plot([x, x], [0.52, 0.68], color="#334155", lw=1.2)
        ax.add_patch(Circle((x, 0.70), 0.0027, facecolor="#C62828", edgecolor="white", linewidth=0.8, zorder=4))
        ax.text(x, -0.11, f"T{index}", ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(x, -0.30, f"{x*1000:.0f}", ha="center", va="top", fontsize=8, color="#64748B")
    for x, label in [(HOT_INTERFACE_M, "hot interface"), (COLD_INTERFACE_M, "cold interface")]:
        ax.axvline(x, color="#9A3412", ls="--", lw=1.35)
        ax.text(x, 0.91, label, ha="center", va="bottom", fontsize=8.5, color="#9A3412")
    ax.add_patch(FancyArrowPatch((-0.011, 0.29), (0.116, 0.29), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#B42318"))
    ax.text(0.052, 0.13, "heat flow", ha="center", va="center", fontsize=10, fontweight="bold", color="#B42318")
    ax.text(0.052, -0.53, "position x (mm)", ha="center", va="center", fontsize=9, color="#64748B")
    fig.tight_layout()
    return fig


def radiation_schematic() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(Rectangle((3.2, 1.0), 3.6, 8.0, facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.6))
    ax.add_patch(Rectangle((3.2, 5.0), 3.6, 2.8, facecolor="#F6B26B", edgecolor="#B45309", linewidth=1.4, alpha=0.78))
    ax.text(7.1, 6.4, "heated duct wall (T10)", va="center", fontsize=10, color="#9A3412", fontweight="bold")
    ax.add_patch(Circle((5.0, 1.7), 0.58, facecolor="#D8E7F0", edgecolor="#0B4F8A", linewidth=1.5))
    ax.text(5.0, 1.7, "FAN", ha="center", va="center", fontsize=8, fontweight="bold", color="#0B4F8A")
    for y in (2.5, 3.3, 4.1):
        ax.add_patch(FancyArrowPatch((5.0, y - 0.25), (5.0, y + 0.25), arrowstyle="-|>", mutation_scale=15, color="#0B72A7", lw=1.5))
    ax.text(2.95, 3.3, "air at T6", ha="right", va="center", fontsize=10, color="#0B4F8A", fontweight="bold")
    bead_x = [4.25, 5.0, 5.78]
    sizes = [0.09, 0.09, 0.23]
    colours = ["#D7DDE3", "#111827", "#111827"]
    labels = ["T7\npolished\nε=0.17", "T8\nsmall black\nε=0.98", "T9\nlarge black\nε=0.98"]
    for x, radius, colour, label in zip(bead_x, sizes, colours, labels):
        ax.add_patch(Circle((x, 6.4), radius, facecolor=colour, edgecolor="#111827", linewidth=1.2, zorder=4))
        ax.plot([x, x], [6.4 + radius, 8.6], color="#475569", lw=1.0)
        ax.text(x, 9.0, label, ha="center", va="top", fontsize=8.5, color="#334155")
    for x in bead_x:
        ax.add_patch(FancyArrowPatch((3.55, 6.4), (x - 0.13, 6.4), arrowstyle="->", mutation_scale=12, color="#C2410C", lw=1.0))
        ax.add_patch(FancyArrowPatch((6.45, 6.4), (x + 0.13, 6.4), arrowstyle="->", mutation_scale=12, color="#C2410C", lw=1.0))
    ax.add_patch(Rectangle((3.8, 5.65), 2.4, 1.5, fill=False, edgecolor="#0F766E", linewidth=2.0, linestyle="--"))
    ax.text(7.1, 5.45, "movable radiation shield", va="center", fontsize=10, color="#0F766E", fontweight="bold")
    ax.text(7.1, 4.95, "raised = shielded\nlowered = exposed", va="top", fontsize=9, color="#536779")
    ax.text(5.0, 0.45, "HT16C concept schematic", ha="center", va="center", fontsize=9, color="#64748B")
    fig.tight_layout()
    return fig


def render_prepare(practical: str) -> None:
    section_heading(
        "Prepare",
        "Understand the purpose, enter your details and watch the procedure before touching the apparatus.",
    )
    if practical == PRACTICAL_1:
        big = "A perfect temperature line becomes three lines with two jumps."
        text = (
            "Inside each solid section, Fourier conduction produces an approximately linear temperature profile. "
            "At a real interface, microscopic roughness leaves only a fraction of the apparent area in true contact. "
            "The resulting contact resistance requires a finite temperature drop to carry the heat."
        )
        outcomes = [
            ("Measure", "Temperature distribution", "Record T1-T8 only after steady state and preserve the raw readings."),
            ("Calculate", "k and contact resistance", "Use regional fits, interface extrapolation and q'' = VI/A."),
            ("Explain", "Why the jumps matter", "Separate material resistance from interface resistance and identify assumptions."),
        ]
    else:
        big = "A thermocouple measures its own temperature, not automatically the air temperature."
        text = (
            "Convection pulls the bead toward the air temperature while radiation pulls it toward the temperature of the surfaces it sees. "
            "The stable reading is the energy-balance result. A shield, lower emissivity or stronger airflow can reduce the bias."
        )
        outcomes = [
            ("Measure", "Four operating cases", "Compare natural/forced convection with exposed/shielded sensors."),
            ("Calculate", "Sensor bias and correction", "Use convection-radiation balance and compare corrected Tm with T6."),
            ("Explain", "Sensor design choices", "Relate error to emissivity, bead diameter, airflow and view of hot walls."),
        ]
    st.markdown(
        f'<div class="concept-card"><div class="label">The central idea</div><div class="big">{html_escape(big)}</div><div class="text">{html_escape(text)}</div></div>',
        unsafe_allow_html=True,
    )
    info_cards(outcomes)

    left, right = st.columns([0.95, 1.25], gap="large")
    with left:
        st.subheader("Student and session details")
        st.text_input("Full name *", key="student_name", placeholder="Enter your name")
        st.text_input("JCU student ID *", key="student_id", placeholder="8 digits", max_chars=8)
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Group / bench", key="group", placeholder="e.g. Group B")
        with c2:
            st.date_input("Laboratory date", key="lab_date")
        st.radio(
            "Practical pathway",
            ["Physical laboratory", "Demonstration data exploration"],
            key="pathway",
            help="Demonstration data are for learning and app testing only; they are not your experimental results.",
        )
        if st.session_state.student_id and not JCU_ID_PATTERN.fullmatch(st.session_state.student_id.strip()):
            st.warning("The JCU student ID should contain exactly 8 digits.")
        elif identity_complete():
            st.markdown('<div class="status-strip success"><b>Details complete.</b> Your entries are being held in this browser session.</div>', unsafe_allow_html=True)
    with right:
        st.subheader("Instruction video")
        st.video(VIDEO_URLS[practical])
        prefix = practical_prefix(practical)
        label = (
            "I have watched the procedure and know which readings must be recorded."
            if st.session_state.pathway == "Demonstration data exploration"
            else "I have watched the procedure or attended the demonstrator briefing."
        )
        st.checkbox(label, key=f"{prefix}_video_ready")
        st.markdown(
            '<div class="status-strip info"><b>Before the lab:</b> complete Prepare and Predict. During the lab: use Apparatus and Record data. After the lab: finish Calculate through Review.</div>',
            unsafe_allow_html=True,
        )
        st.caption("The app supports analysis and understanding. Submit the report through the method specified in LearnJCU.")


COND_QUIZ = [
    (
        "What is the clearest signature of thermal contact resistance on a temperature-position graph?",
        ["A vertical temperature jump at the interface", "A zero temperature gradient everywhere", "A larger bar diameter", "An oscillating temperature"],
        "A vertical temperature jump at the interface",
        "Heat flux remains continuous, but an interface resistance requires a finite temperature difference: ΔTcontact = q''R''c.",
    ),
    (
        "If heat flux doubles while the interface condition is unchanged, what happens to the contact temperature jump?",
        ["It approximately doubles", "It halves", "It becomes zero", "It is unrelated to heat flux"],
        "It approximately doubles",
        "For a fixed area-specific contact resistance, ΔTcontact is proportional to heat flux.",
    ),
    (
        "Which unit belongs to area-specific thermal contact resistance R''c?",
        ["m²·K/W", "W/(m·K)", "W/m²", "K/W²"],
        "m²·K/W",
        "R''c = ΔT/q'', so its unit is K divided by W/m², or m²·K/W.",
    ),
    (
        "Why are lines fitted separately to the hot bar, sample and cold bar?",
        ["To extrapolate each material to the interfaces", "To force every thermocouple onto one line", "To remove the contact jumps", "To avoid using electrical power"],
        "To extrapolate each material to the interfaces",
        "The two fitted temperatures at one interface describe the temperatures on its two sides; their difference is the contact jump.",
    ),
]

RAD_QUIZ = [
    (
        "A thermocouple in cool air sees a much hotter wall. With no shield, its reading will usually be:",
        ["Higher than the true air temperature", "Exactly the air temperature", "Lower than both the air and wall", "Independent of emissivity"],
        "Higher than the true air temperature",
        "Radiation from the hot wall warms the bead until convection back to the cooler air balances the radiative gain.",
    ),
    (
        "Which surface is most sensitive to radiation from the hot wall?",
        ["High-emissivity black surface", "Low-emissivity polished surface", "Both are always identical", "Emissivity affects only conduction"],
        "High-emissivity black surface",
        "A high-emissivity surface is also a strong absorber, so radiative exchange has more influence on its equilibrium temperature.",
    ),
    (
        "Why does stronger airflow generally reduce radiation-induced temperature error?",
        ["It increases convective coupling to the air", "It makes emissivity equal to zero", "It raises the wall temperature", "It blocks all radiation"],
        "It increases convective coupling to the air",
        "A larger h pulls the bead temperature more strongly toward the air temperature.",
    ),
    (
        "What does a radiation shield mainly change?",
        ["The surfaces seen by the thermocouple", "The Stefan-Boltzmann constant", "The thermocouple voltage scale", "The air molecular weight"],
        "The surfaces seen by the thermocouple",
        "The shield intercepts the bead's view of the hot duct wall and presents a less extreme radiative environment.",
    ),
]


def render_predict(practical: str) -> None:
    section_heading(
        "Predict",
        "Commit to a physical prediction before seeing the calculated answer. This makes the practical an experiment rather than a data-entry exercise.",
    )
    quiz = COND_QUIZ if practical == PRACTICAL_1 else RAD_QUIZ
    prefix = practical_prefix(practical)
    st.markdown(
        '<div class="status-strip info"><b>Prediction rule:</b> answer from your present understanding. Incorrect predictions are useful when you later explain what changed your mind.</div>',
        unsafe_allow_html=True,
    )
    for index, (question, options, _, _) in enumerate(quiz, start=1):
        st.radio(question, options, index=None, key=f"{prefix}_quiz_{index}")
    if st.button("Check my predictions", type="primary", key=f"{prefix}_quiz_submit"):
        unanswered = [index for index in range(1, len(quiz) + 1) if not st.session_state.get(f"{prefix}_quiz_{index}")]
        if unanswered:
            st.warning("Answer every prediction before checking: " + ", ".join(str(index) for index in unanswered) + ".")
        else:
            st.session_state[f"{prefix}_quiz_submitted"] = True
    if st.session_state.get(f"{prefix}_quiz_submitted"):
        score = 0
        st.subheader("Feedback")
        for index, (_, _, answer, explanation) in enumerate(quiz, start=1):
            response = st.session_state.get(f"{prefix}_quiz_{index}")
            correct = response == answer
            score += int(correct)
            label = "Correct" if correct else f"Review - correct answer: {answer}"
            css = "success" if correct else "warning"
            st.markdown(
                f'<div class="status-strip {css}"><b>{index}. {html_escape(label)}</b><br>{html_escape(explanation)}</div>',
                unsafe_allow_html=True,
            )
        metric_cards([("Prediction score", f"{score}/{len(quiz)}", "Use the explanations, not only the score")])


COND_SAFETY_ITEMS = [
    "I am wearing fully enclosed footwear and long hair is secured.",
    "I checked cooling-water tubing and connections before opening the flow valve.",
    "I will not change or unclamp a specimen while the heater is energised or hot.",
    "I will keep water away from the electrical service unit and report leaks immediately.",
    "I will wait for stable readings and record the raw values before processing them.",
]

RAD_SAFETY_ITEMS = [
    "I am wearing fully enclosed footwear and long hair is secured.",
    "I will not touch the electrically heated duct or reach into the apparatus.",
    "I will keep loose items clear of the fan inlet and moving anemometer.",
    "I will change the shield and fan only as directed by the demonstrator.",
    "I will wait for stable readings and record T6-T10 for every operating case.",
]


def render_apparatus(practical: str) -> None:
    section_heading(
        "Apparatus and procedure",
        "Connect each sensor number to a physical location, pass the safety gate and collect only steady-state measurements.",
    )
    if practical == PRACTICAL_1:
        st.pyplot(conduction_schematic(), use_container_width=True)
        info_cards(
            [
                ("Sensors", "T1-T3: hot bar", "Three points define the hot-section line that is extrapolated to x = 37.5 mm."),
                ("Sample", "T4-T5: specimen", "The 15 mm spacing gives the sample gradient used in Fourier's law."),
                ("Sensors", "T6-T8: cold bar", "Three points define the cold-section line extrapolated back to x = 67.5 mm."),
            ]
        )
        with st.expander("Guided procedure", expanded=True):
            st.markdown(
                """
                1. With the heater off and apparatus cool, clamp the selected intermediate specimen between the heated and cooled sections.
                2. Inspect the cooling-water connections, start the water and set the flow to approximately 1.5 L/min or the demonstrator's value.
                3. Set the first heater voltage. Monitor temperatures until the system is steady.
                4. Record voltage, current, water flow and T1-T8 without rounding away instrument resolution.
                5. Repeat at the other requested heat inputs, then de-energise and cool before changing the specimen.
                """
            )
            st.markdown(
                '<div class="status-strip warning"><b>Steady-state decision:</b> do not use “ten minutes” as the only test. Confirm that key temperatures show negligible drift over the observation interval specified by your demonstrator.</div>',
                unsafe_allow_html=True,
            )
        safety_items = COND_SAFETY_ITEMS
        safety_key = "cond_safety"
    else:
        st.pyplot(radiation_schematic(), use_container_width=True)
        info_cards(
            [
                ("Reference", "T6: actual air", "Located upstream of the heated measurement section; use it as the comparison air temperature."),
                ("Test beads", "T7, T8 and T9", "Polished 0.5 mm, black 0.5 mm and black 3 mm beads respond differently."),
                ("Radiative source", "T10: heated wall", "The wall temperature sets the radiative environment seen by exposed beads."),
            ]
        )
        with st.expander("Guided procedure", expanded=True):
            st.markdown(
                """
                1. Identify T6-T10, the fan control and the movable radiation shield.
                2. Set the heater voltage to the instructed value (the supplied notes specify 12 V) and wait for stable temperatures.
                3. Record natural-convection readings with the shield lowered (beads exposed), then raised (beads shielded).
                4. Lower the shield, set air velocity to approximately 4 m/s and record the forced-convection exposed case.
                5. Raise the shield and record the forced-convection shielded case after the readings stabilise.
                """
            )
            st.markdown(
                '<div class="status-strip warning"><b>Case labels matter:</b> write down fan state, shield state and air velocity with every temperature set. A correct number attached to the wrong case produces a misleading conclusion.</div>',
                unsafe_allow_html=True,
            )
        safety_items = RAD_SAFETY_ITEMS
        safety_key = "rad_safety"
    st.subheader("Safety and data-quality gate")
    completed = {}
    for index, item in enumerate(safety_items, start=1):
        key = f"{safety_key}_{index}"
        completed[str(index)] = st.checkbox(item, key=key)
    st.session_state[safety_key] = completed
    if all(completed.values()):
        st.markdown('<div class="status-strip success"><b>Gate complete.</b> Follow demonstrator instructions at all times.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-strip warning"><b>Gate incomplete.</b> Confirm every item before operating the apparatus.</div>', unsafe_allow_html=True)


def replace_conduction_data(data: pd.DataFrame) -> None:
    st.session_state.conduction_data = data.copy()
    st.session_state.cond_editor_version += 1


def replace_radiation_data(data: pd.DataFrame) -> None:
    st.session_state.radiation_data = data.copy()
    st.session_state.rad_editor_version += 1


def render_conduction_record() -> None:
    st.markdown(
        '<div class="concept-card"><div class="label">Record first, calculate second</div><div class="big">Preserve all eight temperatures as one operating point.</div><div class="text">T1-T8, voltage and current must belong to the same stable condition. Do not mix readings taken before and after a control change.</div></div>',
        unsafe_allow_html=True,
    )
    b1, b2, spacer = st.columns([1, 1, 2.2])
    with b1:
        if st.button("Load demonstration data", use_container_width=True, help="Loads values transcribed from the supplied 2021 sample report."):
            replace_conduction_data(demonstration_conduction_data())
            st.rerun()
    with b2:
        if st.button("Reset to blank table", use_container_width=True):
            replace_conduction_data(blank_conduction_data())
            st.rerun()
    if st.session_state.pathway == "Physical laboratory":
        st.caption("For assessed work, enter only your group's measured values. Demonstration data are clearly labelled and must not be presented as your measurements.")
    column_config = {
        "Material": st.column_config.SelectboxColumn("Material", options=["Brass", "Aluminium", "Other"], required=True),
        "Trial": st.column_config.TextColumn("Trial"),
        "Voltage_V": st.column_config.NumberColumn("V (V)", min_value=0.0, format="%.3f"),
        "Current_A": st.column_config.NumberColumn("I (A)", min_value=0.0, format="%.3f"),
        "Water_flow_L_min": st.column_config.NumberColumn("Fw (L/min)", min_value=0.0, format="%.3f"),
    }
    for index in range(1, 9):
        column_config[f"T{index}_C"] = st.column_config.NumberColumn(f"T{index} (°C)", format="%.2f")
    edited = st.data_editor(
        st.session_state.conduction_data,
        key=f"cond_editor_{st.session_state.cond_editor_version}",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=330,
    )
    st.session_state.conduction_data = edited.copy()
    valid = valid_conduction_rows(edited)
    incomplete = len(edited) - len(valid)
    if len(valid):
        st.markdown(
            f'<div class="status-strip success"><b>{len(valid)} complete operating point(s).</b> These rows are ready for calculation.</div>',
            unsafe_allow_html=True,
        )
        if len(valid) < 4:
            st.caption(f"The full workflow expects at least 4 complete operating points; {4-len(valid)} more are needed for stage completion.")
    if incomplete:
        st.markdown(
            f'<div class="status-strip warning"><b>{incomplete} incomplete row(s).</b> A row needs V, I and T1-T8 before it can be analysed.</div>',
            unsafe_allow_html=True,
        )
    with st.expander("Data dictionary and quick checks"):
        st.markdown(
            """
            - **T1-T3:** heated brass section, 15 mm spacing.
            - **T4-T5:** intermediate specimen, 15 mm spacing.
            - **T6-T8:** cooled brass section, 15 mm spacing.
            - **Electrical heat input:** the basic analysis uses Q = VI; the fraction reaching one-dimensional conduction is an explicit assumption adjusted in Calculate.
            - **Expected trend:** for heat flowing from T1 toward T8, the temperatures should generally decrease with x. Small measurement scatter is possible; a reversal needs investigation.
            """
        )


def render_radiation_record() -> None:
    st.markdown(
        '<div class="concept-card"><div class="label">Four controlled comparisons</div><div class="big">Change one physical influence at a time.</div><div class="text">Cases 1-2 isolate the shield effect under natural convection. Cases 3-4 repeat the shield comparison under forced convection. Matching case labels is essential.</div></div>',
        unsafe_allow_html=True,
    )
    b1, b2, spacer = st.columns([1, 1, 2.2])
    with b1:
        if st.button("Load demonstration data", use_container_width=True, key="load_rad_demo", help="Loads values transcribed from the supplied 2021 sample report."):
            replace_radiation_data(demonstration_radiation_data())
            st.rerun()
    with b2:
        if st.button("Reset to blank table", use_container_width=True, key="reset_rad"):
            replace_radiation_data(blank_radiation_data())
            st.rerun()
    if st.session_state.pathway == "Physical laboratory":
        st.caption("For assessed work, enter only your group's measured values. Demonstration data are clearly labelled and must not be presented as your measurements.")
    column_config = {
        "Case": st.column_config.TextColumn("Operating case", required=True, width="large"),
        "Fan": st.column_config.SelectboxColumn("Fan", options=["Off", "On"], required=True),
        "Shield": st.column_config.SelectboxColumn("Shield", options=["Down (exposed)", "Up (shielded)"], required=True, width="medium"),
        "Air_velocity_m_s": st.column_config.NumberColumn("Air speed (m/s)", min_value=0.0, format="%.2f"),
        "T6_air_C": st.column_config.NumberColumn("T6 air (°C)", format="%.2f"),
        "T7_polished_C": st.column_config.NumberColumn("T7 polished (°C)", format="%.2f"),
        "T8_small_black_C": st.column_config.NumberColumn("T8 small black (°C)", format="%.2f"),
        "T9_large_black_C": st.column_config.NumberColumn("T9 large black (°C)", format="%.2f"),
        "T10_wall_C": st.column_config.NumberColumn("T10 wall (°C)", format="%.2f"),
    }
    edited = st.data_editor(
        st.session_state.radiation_data,
        key=f"rad_editor_{st.session_state.rad_editor_version}",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=300,
    )
    st.session_state.radiation_data = edited.copy()
    valid = valid_radiation_rows(edited)
    incomplete = len(edited) - len(valid)
    if len(valid):
        st.markdown(
            f'<div class="status-strip success"><b>{len(valid)} complete case(s).</b> These rows are ready for calculation.</div>',
            unsafe_allow_html=True,
        )
        if len(valid) < 4:
            st.caption(f"The full workflow expects all 4 operating cases; {4-len(valid)} more are needed for stage completion.")
    if incomplete:
        st.markdown(
            f'<div class="status-strip warning"><b>{incomplete} incomplete row(s).</b> Each case needs air speed and T6-T10.</div>',
            unsafe_allow_html=True,
        )
    with st.expander("Sensor dictionary and quick checks"):
        st.markdown(
            """
            - **T6:** reference air temperature before the heated test section.
            - **T7:** 0.5 mm polished bead, emissivity approximately 0.17.
            - **T8:** 0.5 mm black bead, emissivity approximately 0.98.
            - **T9:** 3 mm black bead, emissivity approximately 0.98.
            - **T10:** heated duct wall temperature used as the surrounding radiative-source temperature for an exposed-bead model.
            - **Consistency check:** the wall should normally be hotter than the air, and exposed high-emissivity beads should be especially sensitive to it.
            """
        )


def render_record(practical: str) -> None:
    section_heading(
        "Record data",
        "Enter raw measurements with units and operating conditions. The app keeps incomplete rows but excludes them from calculations.",
    )
    if practical == PRACTICAL_1:
        render_conduction_record()
    else:
        render_radiation_record()


def conduction_temperature_plot(raw_row: pd.Series, result: pd.Series) -> plt.Figure:
    temperatures = np.array([float(raw_row[f"T{i}_C"]) for i in range(1, 9)])
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.scatter(THERMOCOUPLE_POSITIONS_M * 1000, temperatures, s=68, color="#0B4F8A", edgecolor="white", linewidth=1.2, zorder=5, label="Measured thermocouples")
    regions = [
        (np.linspace(0.000, HOT_INTERFACE_M, 60), result["Hot_slope_K_m"], result["Hot_intercept_C"], "Hot-bar fit", "#C2410C"),
        (np.linspace(HOT_INTERFACE_M, COLD_INTERFACE_M, 60), result["Sample_slope_K_m"], result["Sample_intercept_C"], "Sample fit", "#B28A00"),
        (np.linspace(COLD_INTERFACE_M, 0.105, 60), result["Cold_slope_K_m"], result["Cold_intercept_C"], "Cold-bar fit", "#0F766E"),
    ]
    for x, slope, intercept, label, colour in regions:
        ax.plot(x * 1000, slope * x + intercept, lw=2.4, color=colour, label=label)
    for x, label in [(HOT_INTERFACE_M, "hot interface"), (COLD_INTERFACE_M, "cold interface")]:
        ax.axvline(x * 1000, color="#94A3B8", ls="--", lw=1.2)
        ax.text(x * 1000, ax.get_ylim()[1] if ax.get_ylim() else max(temperatures), label, ha="center", va="top", fontsize=8.5, color="#64748B")
    ax.plot(
        [HOT_INTERFACE_M * 1000, HOT_INTERFACE_M * 1000],
        [result["Hot_bar_face_C"], result["Sample_hot_face_C"]],
        color="#B42318",
        lw=4,
        solid_capstyle="round",
        label="Contact temperature jump",
    )
    ax.plot(
        [COLD_INTERFACE_M * 1000, COLD_INTERFACE_M * 1000],
        [result["Sample_cold_face_C"], result["Cold_bar_face_C"]],
        color="#B42318",
        lw=4,
        solid_capstyle="round",
    )
    for index, (x, y) in enumerate(zip(THERMOCOUPLE_POSITIONS_M * 1000, temperatures), start=1):
        ax.annotate(f"T{index}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color="#334155")
    ax.set_xlabel("Position x (mm)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Measured distribution and regional extrapolations", loc="left", fontweight="bold", color="#183A57")
    ax.grid(alpha=0.20)
    ax.legend(ncol=3, frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()
    return fig


def conduction_resistance_plot(result: pd.Series) -> plt.Figure:
    labels = ["Hot contact", "Sample", "Cold contact"]
    drops = np.array([
        result["Hot_contact_jump_K"],
        result["Sample_drop_K"],
        result["Cold_contact_jump_K"],
    ])
    colours = ["#C2410C", "#D4A72C", "#0F766E"]
    fig, ax = plt.subplots(figsize=(9.0, 3.1))
    left = 0.0
    for label, drop, colour in zip(labels, drops, colours):
        width = max(float(drop), 0.0)
        ax.barh([0], [width], left=left, color=colour, height=0.42, label=label)
        if width > 0:
            ax.text(left + width / 2, 0, f"{drop:.2f} K", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += width
    ax.set_xlabel("Temperature drop carried by each resistance (K)")
    ax.set_yticks([])
    ax.set_title("Where the measured temperature difference is spent", loc="left", fontweight="bold", color="#183A57")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.80, bottom=0.36)
    return fig


def render_conduction_calculate() -> None:
    valid = valid_conduction_rows(st.session_state.conduction_data)
    if valid.empty:
        st.warning("Enter at least one complete operating point in Record data before calculating.")
        return
    st.subheader("Analysis assumptions")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.number_input(
            "Bar / interface diameter (mm)",
            min_value=1.0,
            max_value=100.0,
            step=0.1,
            key="diameter_mm",
            help="The supplied apparatus geometry is consistent with a 25 mm interface diameter. Confirm your apparatus.",
        )
    with a2:
        st.number_input(
            "Fraction of VI assumed to conduct through the bar",
            min_value=0.10,
            max_value=1.00,
            step=0.01,
            key="heat_fraction",
            help="1.00 is the simple adiabatic assumption Q = VI. Lower values allow sensitivity testing for environmental heat loss.",
        )
    with a3:
        st.number_input(
            "Brass reference k (W/m·K)",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="brass_reference_k",
            help="The supplied sample report quotes 110-128 W/(m·K); 119 is the midpoint. Confirm the required reference for the actual alloy and temperature.",
        )
    with a4:
        st.number_input(
            "Aluminium reference k (W/m·K)",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="aluminium_reference_k",
            help="The supplied sample report uses approximately 180 W/(m·K). Confirm the actual alloy and temperature.",
        )
    analysed = analyse_conduction(valid, st.session_state.diameter_mm, st.session_state.heat_fraction)
    if analysed.empty:
        st.error("The current assumptions do not permit analysis.")
        return
    analysed["Reference_k_W_mK"] = analysed["Material"].map(
        {"Brass": st.session_state.brass_reference_k, "Aluminium": st.session_state.aluminium_reference_k}
    )
    analysed["Deviation_from_reference_pct"] = 100.0 * (
        analysed["Thermal_conductivity_W_mK"] - analysed["Reference_k_W_mK"]
    ) / analysed["Reference_k_W_mK"]
    labels = [
        f"{row.Material} · trial {row.Trial} · {row.Voltage_V:.2f} V"
        for row in analysed.itertuples(index=False)
    ]
    selected_label = st.selectbox("Operating point to examine", labels)
    selected_index = labels.index(selected_label)
    result = analysed.iloc[selected_index]
    raw_row = valid.loc[int(result["Source_row"])]

    reference_value = result["Reference_k_W_mK"]
    deviation_value = result["Deviation_from_reference_pct"]
    metric_cards(
        [
            ("Sample conductivity", f"{result['Thermal_conductivity_W_mK']:.1f} W/m·K", "From the T4-T5 gradient"),
            ("Difference from reference", f"{deviation_value:+.1f}%" if pd.notna(deviation_value) else "Not set", f"Reference: {reference_value:.1f} W/m·K" if pd.notna(reference_value) else "Choose Brass or Aluminium"),
            ("Hot contact R''", f"{result['Hot_contact_Rpp_m2K_W']:.2e}", "m²·K/W"),
            ("Cold contact R''", f"{result['Cold_contact_Rpp_m2K_W']:.2e}", "m²·K/W"),
        ]
    )
    st.markdown(
        f'<div class="concept-card"><div class="label">What the fit reveals</div><div class="big">{result["Contact_share_pct"]:.1f}% of the fitted temperature drop occurs at the two contacts.</div><div class="text">The sample is not the only thermal resistance. In this operating point, the fitted hot-contact jump is {result["Hot_contact_jump_K"]:.2f} K and the cold-contact jump is {result["Cold_contact_jump_K"]:.2f} K. Treat unusually large or negative jumps as evidence to investigate, not as values to accept automatically.</div></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(conduction_temperature_plot(raw_row, result), use_container_width=True)
    st.pyplot(conduction_resistance_plot(result), use_container_width=True)

    summary_columns = [
        "Material",
        "Trial",
        "Voltage_V",
        "Electrical_power_W",
        "Thermal_conductivity_W_mK",
        "Reference_k_W_mK",
        "Deviation_from_reference_pct",
        "Hot_contact_Rpp_m2K_W",
        "Cold_contact_Rpp_m2K_W",
        "Contact_share_pct",
        "Quality_flags",
    ]
    st.subheader("All operating points")
    st.dataframe(
        analysed[summary_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Voltage_V": st.column_config.NumberColumn("Voltage (V)", format="%.2f"),
            "Electrical_power_W": st.column_config.NumberColumn("VI (W)", format="%.3f"),
            "Thermal_conductivity_W_mK": st.column_config.NumberColumn("k (W/m·K)", format="%.1f"),
            "Reference_k_W_mK": st.column_config.NumberColumn("Reference k (W/m·K)", format="%.1f"),
            "Deviation_from_reference_pct": st.column_config.NumberColumn("Difference (%)", format="%+.1f"),
            "Hot_contact_Rpp_m2K_W": st.column_config.NumberColumn("Hot R'' (m²K/W)", format="%.3e"),
            "Cold_contact_Rpp_m2K_W": st.column_config.NumberColumn("Cold R'' (m²K/W)", format="%.3e"),
            "Contact_share_pct": st.column_config.NumberColumn("Contact share (%)", format="%.1f"),
        },
    )
    if result["Quality_flags"] == "No automatic flags":
        st.markdown('<div class="status-strip success"><b>Automatic checks passed.</b> The temperature order, fitted linearity and contact-jump signs are plausible.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-strip warning"><b>Investigate:</b> {html_escape(result["Quality_flags"])}</div>', unsafe_allow_html=True)

    with st.expander("Worked calculation for the selected operating point", expanded=True):
        area = result["Area_m2"]
        st.latex(r"A=\frac{\pi D^2}{4}")
        st.write(f"A = π({st.session_state.diameter_mm/1000:.4f} m)²/4 = {area:.6e} m²")
        st.latex(r"Q=f_QVI,\qquad q''=\frac{Q}{A}")
        st.write(
            f"Q = {st.session_state.heat_fraction:.2f} × {result['Voltage_V']:.3f} × {result['Current_A']:.3f} "
            f"= {result['Assumed_conduction_heat_W']:.3f} W; q'' = {result['Heat_flux_W_m2']:.1f} W/m²"
        )
        st.latex(r"k=-\frac{q''}{dT/dx}")
        st.write(
            f"The T4-T5 fit gives dT/dx = {result['Sample_slope_K_m']:.2f} K/m, so k = {result['Thermal_conductivity_W_mK']:.2f} W/(m·K)."
        )
        st.latex(r"R''_c=\frac{\Delta T_{interface}}{q''}")
        st.write(
            f"Hot contact: {result['Hot_contact_jump_K']:.2f} K / {result['Heat_flux_W_m2']:.1f} W/m² = {result['Hot_contact_Rpp_m2K_W']:.3e} m²·K/W."
        )
        st.write(
            f"Cold contact: {result['Cold_contact_jump_K']:.2f} K / {result['Heat_flux_W_m2']:.1f} W/m² = {result['Cold_contact_Rpp_m2K_W']:.3e} m²·K/W."
        )

    with st.expander("Quantitative uncertainty estimate for k"):
        c1, c2, c3, c4, c5 = st.columns(5)
        d_v = c1.number_input("±V (V)", 0.0, 5.0, 0.01, 0.01, key="unc_v")
        d_i = c2.number_input("±I (A)", 0.0, 2.0, 0.01, 0.01, key="unc_i")
        d_t = c3.number_input("±T (K)", 0.0, 5.0, 0.10, 0.05, key="unc_t")
        d_d = c4.number_input("±D (mm)", 0.0, 5.0, 0.10, 0.05, key="unc_d")
        d_l = c5.number_input("±Δx (mm)", 0.0, 5.0, 0.10, 0.05, key="unc_l")
        delta_t_sample = float(raw_row["T4_C"] - raw_row["T5_C"])
        uncertainty = conduction_uncertainty_percent(
            result["Voltage_V"],
            result["Current_A"],
            delta_t_sample,
            st.session_state.diameter_mm,
            15.0,
            d_v,
            d_i,
            d_t,
            d_d,
            d_l,
        )
        if math.isfinite(uncertainty):
            st.write(f"Estimated RSS relative uncertainty in k: **{uncertainty:.1f}%** (instrument uncertainties only).")
            if abs(delta_t_sample) < 2 * math.sqrt(2) * d_t:
                st.warning("The T4-T5 temperature difference is small relative to the thermocouple uncertainty; k will be highly sensitive to small reading changes.")
        st.caption("This does not include systematic effects such as lateral heat loss, imperfect steady state, geometry error or the assumption Q = VI.")


def radiation_error_plot(analysed: pd.DataFrame) -> plt.Figure:
    labels = [str(value) for value in analysed["Case"]]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    series = [
        ("T7 polished", analysed["T7_error_K"].to_numpy(), "#7A8896"),
        ("T8 small black", analysed["T8_error_K"].to_numpy(), "#C2410C"),
        ("T9 large black", analysed["T9_error_K"].to_numpy(), "#0B4F8A"),
    ]
    for offset, (label, values, colour) in zip((-width, 0, width), series):
        bars = ax.bar(x + offset, values, width, label=label, color=colour)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_ylabel("Measurement error Tbead - T6 (K)")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("Radiation-induced bias across the operating cases", loc="left", fontweight="bold", color="#183A57")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def render_radiation_calculate() -> None:
    analysed = analyse_radiation(st.session_state.radiation_data)
    if analysed.empty:
        st.warning("Enter at least one complete operating case in Record data before calculating.")
        return
    st.pyplot(radiation_error_plot(analysed), use_container_width=True)
    display_columns = [
        "Case",
        "Fan",
        "Shield",
        "Air_velocity_m_s",
        "T7_error_K",
        "T8_error_K",
        "T9_error_K",
        "Mean_sensor_error_K",
        "Maximum_abs_error_K",
    ]
    st.dataframe(
        analysed[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Air_velocity_m_s": st.column_config.NumberColumn("Air speed (m/s)", format="%.2f"),
            "T7_error_K": st.column_config.NumberColumn("T7 error (K)", format="%.2f"),
            "T8_error_K": st.column_config.NumberColumn("T8 error (K)", format="%.2f"),
            "T9_error_K": st.column_config.NumberColumn("T9 error (K)", format="%.2f"),
            "Mean_sensor_error_K": st.column_config.NumberColumn("Mean error (K)", format="%.2f"),
            "Maximum_abs_error_K": st.column_config.NumberColumn("Maximum |error| (K)", format="%.2f"),
        },
    )
    worst_index = analysed["Maximum_abs_error_K"].astype(float).idxmax()
    worst = analysed.loc[worst_index]
    metric_cards(
        [
            ("Largest recorded bias", f"{worst['Maximum_abs_error_K']:.2f} K", str(worst["Case"])),
            ("Wall-air difference", f"{(worst['T10_wall_C']-worst['T6_air_C']):.1f} K", "Radiative driving condition in that case"),
            ("T7 polished error", f"{worst['T7_error_K']:.2f} K", "ε ≈ 0.17, diameter 0.5 mm"),
            ("T8 black error", f"{worst['T8_error_K']:.2f} K", "ε ≈ 0.98, diameter 0.5 mm"),
        ]
    )

    st.subheader("Convection-radiation correction")
    labels = [str(value) for value in analysed["Case"]]
    default_index = next((i for i, value in enumerate(analysed["Shield"].astype(str)) if "exposed" in value.lower() and analysed.iloc[i]["Air_velocity_m_s"] > 0), 0)
    selected_label = st.selectbox("Case for the energy-balance calculation", labels, index=default_index)
    row = analysed.iloc[labels.index(selected_label)]
    is_shielded = "shielded" in str(row["Shield"]).lower() and "exposed" not in str(row["Shield"]).lower()
    if is_shielded:
        st.warning("For a raised shield, T10 is not the surface directly seen by the beads. The simple correction using T10 as Tsur is therefore not physically complete. Prefer an exposed case unless shield temperature is measured.")
    velocity = float(row["Air_velocity_m_s"])
    manual_h = None
    if velocity <= 0:
        manual_h = st.number_input("Assumed natural-convection h (W/m²·K)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        st.caption("Natural-convection h is not obtained from the forced-crossflow correlation. Test a plausible range and report the assumption.")
    h_scale = st.slider("h sensitivity multiplier", 0.50, 1.50, 1.00, 0.05, help="Tests sensitivity to property and correlation uncertainty.")
    sensors = [
        ("T7 polished 0.5 mm", float(row["T7_polished_C"]), 0.17, 0.5),
        ("T8 black 0.5 mm", float(row["T8_small_black_C"]), 0.98, 0.5),
        ("T9 black 3 mm", float(row["T9_large_black_C"]), 0.98, 3.0),
    ]
    correction_rows = []
    for name, t_bead, emissivity, diameter in sensors:
        if manual_h is None:
            h_base, reynolds, nusselt = forced_convection_h(velocity, diameter)
        else:
            h_base, reynolds, nusselt = float(manual_h), np.nan, np.nan
        h = h_base * h_scale
        corrected = radiation_corrected_medium_temperature_C(t_bead, float(row["T10_wall_C"]), h, emissivity)
        h_r = linearised_radiation_coefficient_W_m2K(t_bead, float(row["T10_wall_C"]), emissivity)
        correction_rows.append(
            {
                "Sensor": name,
                "Measured_bead_C": t_bead,
                "Emissivity": emissivity,
                "h_W_m2K": h,
                "h_r_W_m2K": h_r,
                "h_r_over_h": h_r / h,
                "Corrected_Tm_C": corrected,
                "Reference_T6_C": float(row["T6_air_C"]),
                "Residual_to_T6_K": corrected - float(row["T6_air_C"]),
                "Re": reynolds,
                "Nu": nusselt,
            }
        )
    correction = pd.DataFrame(correction_rows)
    st.markdown('<div class="equation-box"><b>Energy balance:</b> h(Tm - Tth) = εσ(Tth⁴ - Tsur⁴), therefore Tm = Tth + (εσ/h)(Tth⁴ - Tsur⁴). All temperatures inside the fourth-power term must be in kelvin.</div>', unsafe_allow_html=True)
    st.dataframe(
        correction,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Measured_bead_C": st.column_config.NumberColumn("Measured bead (°C)", format="%.2f"),
            "Emissivity": st.column_config.NumberColumn("ε", format="%.2f"),
            "h_W_m2K": st.column_config.NumberColumn("h (W/m²K)", format="%.1f"),
            "h_r_W_m2K": st.column_config.NumberColumn("hr (W/m²K)", format="%.1f"),
            "h_r_over_h": st.column_config.NumberColumn("hr/h", format="%.3f"),
            "Corrected_Tm_C": st.column_config.NumberColumn("Corrected Tm (°C)", format="%.2f"),
            "Reference_T6_C": st.column_config.NumberColumn("Reference T6 (°C)", format="%.2f"),
            "Residual_to_T6_K": st.column_config.NumberColumn("Residual to T6 (K)", format="%+.2f"),
            "Re": st.column_config.NumberColumn("Re", format="%.1f"),
            "Nu": st.column_config.NumberColumn("Nu", format="%.2f"),
        },
    )
    mean_abs_before = np.mean([abs(item[1] - float(row["T6_air_C"])) for item in sensors])
    mean_abs_after = float(correction["Residual_to_T6_K"].abs().mean())
    st.markdown(
        f'<div class="concept-card"><div class="label">Model check</div><div class="big">Mean absolute difference from T6: {mean_abs_before:.2f} K before correction, {mean_abs_after:.2f} K after correction.</div><div class="text">A smaller residual supports the model but does not prove every assumption. T6 is upstream, T10 may not represent the full radiative surroundings, h comes from a correlation or assumption, and bead conduction through the leads is neglected.</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Where the forced-convection h values come from"):
        st.latex(r"Re_D=\frac{VD}{\nu},\qquad Nu_D=2+0.6Re_D^{1/2}Pr^{1/3},\qquad h=\frac{Nu_D k_{air}}{D}")
        st.write("The app uses a Ranz-Marshall-style small-sphere estimate with default air properties ν = 15.9×10⁻⁶ m²/s, k = 0.0263 W/(m·K), and Pr = 0.707.")
        st.caption("Use the correlation as an engineering estimate, state it in the report and discuss its validity for the actual bead and duct flow.")


def render_calculate(practical: str) -> None:
    section_heading(
        "Calculate and visualise",
        "Turn raw measurements into evidence, keep assumptions visible and treat automatic flags as prompts for engineering judgment.",
    )
    if practical == PRACTICAL_1:
        render_conduction_calculate()
    else:
        render_radiation_calculate()


def render_conduction_explore() -> None:
    st.markdown(
        '<div class="concept-card"><div class="label">Interactive resistance network</div><div class="big">The same heat flux crosses the hot contact, sample and cold contact.</div><div class="text">Each element consumes part of the available temperature difference. Change the controls and watch where the drop moves.</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        q_flux = st.slider("Heat flux q'' (kW/m²)", 2.0, 80.0, 25.0, 1.0)
        r_hot_micro = st.slider("Hot contact R'' (×10⁻⁴ m²·K/W)", 0.0, 20.0, 2.0, 0.2)
        r_cold_micro = st.slider("Cold contact R'' (×10⁻⁴ m²·K/W)", 0.0, 30.0, 8.0, 0.2)
        conductivity = st.slider("Sample conductivity k (W/m·K)", 20, 240, 120, 5)
        sample_length_mm = st.slider("Sample length L (mm)", 10, 60, 30, 1)
    q_flux_si = q_flux * 1000.0
    r_hot = r_hot_micro * 1e-4
    r_cold = r_cold_micro * 1e-4
    r_sample = (sample_length_mm / 1000.0) / conductivity
    drops = np.array([q_flux_si * r_hot, q_flux_si * r_sample, q_flux_si * r_cold])
    total = float(drops.sum())
    contact_share = 100.0 * (drops[0] + drops[2]) / total if total else 0.0
    with right:
        metric_cards(
            [
                ("Hot contact drop", f"{drops[0]:.2f} K", "q''R''hot"),
                ("Sample drop", f"{drops[1]:.2f} K", "q''L/k"),
                ("Cold contact drop", f"{drops[2]:.2f} K", "q''R''cold"),
                ("Contact share", f"{contact_share:.1f}%", "of total ΔT"),
            ]
        )
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        labels = ["Hot contact", "Sample", "Cold contact"]
        colours = ["#C2410C", "#D4A72C", "#0F766E"]
        bars = ax.bar(labels, drops, color=colours)
        ax.bar_label(bars, fmt="%.2f K", padding=3, fontsize=9)
        ax.set_ylabel("Temperature drop (K)")
        ax.set_title("Temperature-drop allocation", loc="left", fontweight="bold", color="#183A57")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    st.markdown(
        '<div class="equation-box"><b>Series network:</b> ΔTtotal = q''(R''hot + L/k + R''cold). A very thin interface can dominate a much thicker solid because its microscopic contact area is small.</div>',
        unsafe_allow_html=True,
    )
    answer = st.radio(
        "Concept check: if q'' doubles and all three resistances stay constant, what happens?",
        ["Every temperature-drop contribution doubles", "Only the sample drop doubles", "Contact jumps disappear", "The total drop halves"],
        index=None,
        key="cond_explore_answer",
    )
    if answer:
        correct = answer == "Every temperature-drop contribution doubles"
        if correct:
            st.success("Correct. Each series drop is q'' multiplied by its own area-specific resistance.")
            if st.checkbox("I can explain why interface jumps do not violate continuity of heat flow.", key="cond_explore_ack"):
                st.session_state.cond_explore_complete = True
        else:
            st.warning("Revisit ΔT = q''R''. The same steady heat flux crosses every element in this one-dimensional series model.")


def render_radiation_explore() -> None:
    st.markdown(
        '<div class="concept-card"><div class="label">Interactive sensor equilibrium</div><div class="big">The bead settles between the air and the surfaces it sees.</div><div class="text">The reading is set by competing convection and radiation. A stable display does not guarantee an unbiased measurement.</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.90, 1.10], gap="large")
    with left:
        air_c = st.slider("True air temperature Tm (°C)", 0.0, 80.0, 25.0, 1.0)
        wall_c = st.slider("Surrounding wall temperature Tsur (°C)", 0.0, 250.0, 120.0, 5.0)
        h = st.slider("Convection coefficient h (W/m²·K)", 2.0, 500.0, 25.0, 2.0)
        emissivity = st.slider("Bead emissivity ε", 0.05, 1.00, 0.98, 0.01)
    bead_c = equilibrium_sensor_temperature_C(air_c, wall_c, h, emissivity)
    error = bead_c - air_c
    h_r = linearised_radiation_coefficient_W_m2K(bead_c, wall_c, emissivity)
    low_eps_bead = equilibrium_sensor_temperature_C(air_c, wall_c, h, 0.17)
    high_h_bead = equilibrium_sensor_temperature_C(air_c, wall_c, min(h * 4, 2000), emissivity)
    with right:
        metric_cards(
            [
                ("Thermocouple reading", f"{bead_c:.2f} °C", "energy-balance temperature"),
                ("Measurement bias", f"{error:+.2f} K", "Tbead - Tair"),
                ("Linearised hr", f"{h_r:.2f} W/m²·K", "radiative coupling"),
                ("hr / h", f"{h_r/h:.3f}", "relative radiation influence"),
            ]
        )
        fig, ax = plt.subplots(figsize=(8.7, 3.8))
        names = ["True air", "Current bead", "Polished ε=0.17", "4× stronger h", "Hot wall"]
        values = [air_c, bead_c, low_eps_bead, high_h_bead, wall_c]
        colours = ["#0B4F8A", "#C2410C", "#7A8896", "#0F766E", "#B42318"]
        bars = ax.barh(names, values, color=colours)
        ax.bar_label(bars, fmt="%.1f °C", padding=3, fontsize=9)
        ax.set_xlabel("Temperature (°C)")
        ax.set_title("What changes the indicated temperature?", loc="left", fontweight="bold", color="#183A57")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    info_cards(
        [
            ("Reduce radiation", "Use a shield", "Replace the bead's view of an extreme hot surface with a less extreme radiative environment."),
            ("Reduce absorption", "Use a polished bead", "Lower emissivity weakens radiative exchange, although surface finish must remain stable and known."),
            ("Strengthen convection", "Increase airflow", "A larger h couples the bead more strongly to the fluid whose temperature is required."),
        ]
    )
    answer = st.radio(
        "Concept check: why can two thermocouples in the same air report different steady temperatures?",
        [
            "Their emissivity, size and convective coupling produce different energy balances",
            "The Stefan-Boltzmann constant changes for each bead",
            "Steady state removes all radiation",
            "Air temperature has no physical meaning",
        ],
        index=None,
        key="rad_explore_answer",
    )
    if answer:
        correct = answer.startswith("Their emissivity")
        if correct:
            st.success("Correct. Each bead reaches the temperature that balances its own convection, radiation and smaller secondary effects such as lead conduction.")
            if st.checkbox("I can explain how a radiation shield and airflow reduce measurement bias.", key="rad_explore_ack"):
                st.session_state.rad_explore_complete = True
        else:
            st.warning("Return to the energy balance. The constants are the same, but bead properties and heat-transfer coefficients differ.")


def render_explore(practical: str) -> None:
    section_heading(
        "Explore the concept",
        "Use the model as a laboratory for cause and effect. Change one variable at a time and explain the direction of the response.",
    )
    if practical == PRACTICAL_1:
        render_conduction_explore()
    else:
        render_radiation_explore()


def conduction_evidence() -> list[str]:
    analysed = analyse_conduction(
        st.session_state.conduction_data,
        st.session_state.diameter_mm,
        st.session_state.heat_fraction,
    )
    if analysed.empty:
        return ["No complete conduction operating points are available yet."]
    valid_k = analysed["Thermal_conductivity_W_mK"].replace([np.inf, -np.inf], np.nan).dropna()
    valid_contact = analysed.dropna(subset=["Contact_share_pct"])
    flags = int((analysed["Quality_flags"] != "No automatic flags").sum())
    statements = []
    if valid_k.empty:
        statements.append(f"{len(analysed)} operating point(s) were analysed, but no finite k value could be calculated.")
    else:
        statements.append(f"{len(analysed)} operating point(s) were analysed; calculated k spans {valid_k.min():.1f}-{valid_k.max():.1f} W/(m·K).")
    if valid_contact.empty:
        statements.append("No positive fitted total temperature drop was available for a contact-share comparison.")
    else:
        max_contact = valid_contact.loc[valid_contact["Contact_share_pct"].idxmax()]
        statements.extend(
            [
                f"The largest fitted contact share is {max_contact['Contact_share_pct']:.1f}% for {max_contact['Material']} trial {max_contact['Trial']}.",
                f"That operating point has hot and cold contact jumps of {max_contact['Hot_contact_jump_K']:.2f} K and {max_contact['Cold_contact_jump_K']:.2f} K.",
            ]
        )
    statements.extend(
        [
            f"{flags} operating point(s) contain at least one automatic quality flag.",
            f"The analysis assumes {st.session_state.heat_fraction:.2f} of VI crosses a {st.session_state.diameter_mm:.1f} mm diameter section by one-dimensional conduction.",
        ]
    )
    return statements


def radiation_evidence() -> list[str]:
    analysed = analyse_radiation(st.session_state.radiation_data)
    if analysed.empty:
        return ["No complete radiation operating cases are available yet."]
    worst = analysed.loc[analysed["Maximum_abs_error_K"].idxmax()]
    best = analysed.loc[analysed["Maximum_abs_error_K"].idxmin()]
    bead_means = {
        "T7 polished": analysed["T7_error_K"].abs().mean(),
        "T8 small black": analysed["T8_error_K"].abs().mean(),
        "T9 large black": analysed["T9_error_K"].abs().mean(),
    }
    ranked = sorted(bead_means.items(), key=lambda item: item[1], reverse=True)
    return [
        f"The largest absolute sensor error is {worst['Maximum_abs_error_K']:.2f} K in {worst['Case']}.",
        f"The smallest case maximum is {best['Maximum_abs_error_K']:.2f} K in {best['Case']}.",
        f"Across the entered cases, {ranked[0][0]} has the largest mean absolute bias ({ranked[0][1]:.2f} K).",
        f"For the worst case, the measured wall-air temperature difference is {worst['T10_wall_C']-worst['T6_air_C']:.2f} K.",
        "A shield comparison is strongest when fan state and air speed are matched; an airflow comparison is strongest when shield state is matched.",
    ]


def render_interpret(practical: str) -> None:
    section_heading(
        "Interpret results",
        "Build an evidence-based discussion. Use measured trends, equations and limitations; avoid claims that the data do not support.",
    )
    evidence = conduction_evidence() if practical == PRACTICAL_1 else radiation_evidence()
    st.markdown('<div class="evidence"><b>Evidence extracted from your current data</b><ul>' + "".join(f"<li>{html_escape(item)}</li>" for item in evidence) + "</ul></div>", unsafe_allow_html=True)
    if practical == PRACTICAL_1:
        prompts = [
            ("1. Describe the temperature-position graph. Where is it approximately linear, and where are the discontinuities?", "cond_interpret_1", "Refer to T1-T3, T4-T5, T6-T8 and both fitted interface temperatures."),
            ("2. How important are the two contacts relative to the sample resistance?", "cond_interpret_2", "Use contact R'', temperature-drop shares and differences between the hot and cold interfaces."),
            ("3. How do k and apparent contact resistance change across heat inputs or materials?", "cond_interpret_3", "Report the trend, then distinguish a real interface-property change from heat loss, small ΔT, clamping and measurement uncertainty."),
            ("4. What limits the validity of Q = VI and the one-dimensional steady model?", "cond_interpret_4", "Consider lateral heat loss, steady-state drift, thermocouple placement, diameter, contact pressure, regression and sensor uncertainty."),
        ]
        caution = "Do not claim that higher voltage simply “overcomes” surface imperfections. Contact conductance can depend on pressure and temperature, but an apparent trend can also come from heat loss or uncertainty."
    else:
        prompts = [
            ("1. Rank the bead errors and identify the operating case with the strongest radiation effect.", "rad_interpret_1", "Quote Tbead - T6 values and relate them to T10 - T6."),
            ("2. What evidence shows the effects of the shield and forced airflow?", "rad_interpret_2", "Use matched comparisons and acknowledge any inconsistent point rather than hiding it."),
            ("3. Explain the roles of emissivity and bead size.", "rad_interpret_3", "Compare T7 with T8 for emissivity, and T8 with T9 for size while noting that h also depends on diameter."),
            ("4. Did the convection-radiation correction recover T6? What assumptions limit the comparison?", "rad_interpret_4", "Discuss h correlation/assumption, effective surrounding temperature, upstream T6, lead conduction and steady state."),
        ]
        caution = "Do not describe the shield as “removing radiation.” It changes the radiative surroundings and usually reduces net exchange with the hot wall; it does not make radiation zero."
    st.markdown(f'<div class="status-strip warning"><b>Scientific-language check:</b> {html_escape(caution)}</div>', unsafe_allow_html=True)
    for label, key, help_text in prompts:
        st.text_area(label, key=key, height=125, help=help_text, placeholder="Write concise notes in your own words and include numerical evidence...")
    filled = sum(bool(st.session_state[key].strip()) for _, key, _ in prompts)
    if filled == len(prompts):
        st.markdown('<div class="status-strip success"><b>Discussion notes complete.</b> Edit them into a coherent report discussion rather than submitting four disconnected answers.</div>', unsafe_allow_html=True)
    else:
        st.caption(f"{filled} of {len(prompts)} discussion prompts completed.")


def markdown_table(data: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    if data.empty:
        return "_No complete rows available._"
    formats = formats or {}
    selected = data[columns].copy()
    headers = [str(column).replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in selected.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered = ""
            elif column in formats:
                rendered = formats[column].format(value)
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_summary_markdown(practical: str) -> str:
    name = st.session_state.student_name.strip() or "Not entered"
    student_id = st.session_state.student_id.strip() or "Not entered"
    header = f"""# ME3512 ThermalLab practical record

- **Practical:** {PRACTICAL_TITLES[practical]}
- **Student:** {name}
- **JCU ID:** {student_id}
- **Group / bench:** {st.session_state.group or 'Not entered'}
- **Laboratory date:** {st.session_state.lab_date}
- **Pathway:** {st.session_state.pathway}
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

> This is an analysis record, not a finished laboratory report. Check all values and write the final report in your own words.

## Core model
"""
    if practical == PRACTICAL_1:
        analysed = analyse_conduction(st.session_state.conduction_data, st.session_state.diameter_mm, st.session_state.heat_fraction)
        if not analysed.empty:
            analysed["Reference_k_W_mK"] = analysed["Material"].map(
                {"Brass": st.session_state.brass_reference_k, "Aluminium": st.session_state.aluminium_reference_k}
            )
            analysed["Deviation_from_reference_pct"] = 100.0 * (
                analysed["Thermal_conductivity_W_mK"] - analysed["Reference_k_W_mK"]
            ) / analysed["Reference_k_W_mK"]
        model = """
- Fourier conduction: `q'' = -k dT/dx`
- Contact resistance: `R''c = ΔTinterface / q''`
- Interface diameter and assumed heat fraction are retained below.

## Analysis assumptions

- Diameter: {:.2f} mm
- Fraction of VI treated as one-dimensional conduction: {:.3f}
- Brass reference k: {:.1f} W/(m·K)
- Aluminium reference k: {:.1f} W/(m·K)

## Analysed operating points

{}

## Evidence prompts

{}

## Discussion notes

1. **Temperature distribution:** {}
2. **Importance of contacts:** {}
3. **Trends across heat inputs/materials:** {}
4. **Model limitations and errors:** {}

## Instruction video

{}
""".format(
            st.session_state.diameter_mm,
            st.session_state.heat_fraction,
            st.session_state.brass_reference_k,
            st.session_state.aluminium_reference_k,
            markdown_table(
                analysed,
                ["Material", "Trial", "Voltage_V", "Electrical_power_W", "Thermal_conductivity_W_mK", "Reference_k_W_mK", "Deviation_from_reference_pct", "Hot_contact_Rpp_m2K_W", "Cold_contact_Rpp_m2K_W", "Contact_share_pct", "Quality_flags"],
                {"Voltage_V": "{:.2f}", "Electrical_power_W": "{:.3f}", "Thermal_conductivity_W_mK": "{:.2f}", "Reference_k_W_mK": "{:.1f}", "Deviation_from_reference_pct": "{:+.1f}", "Hot_contact_Rpp_m2K_W": "{:.3e}", "Cold_contact_Rpp_m2K_W": "{:.3e}", "Contact_share_pct": "{:.1f}"},
            ),
            "\n".join(f"- {item}" for item in conduction_evidence()),
            st.session_state.cond_interpret_1 or "Not completed",
            st.session_state.cond_interpret_2 or "Not completed",
            st.session_state.cond_interpret_3 or "Not completed",
            st.session_state.cond_interpret_4 or "Not completed",
            VIDEO_URLS[practical],
        )
    else:
        analysed = analyse_radiation(st.session_state.radiation_data)
        model = """
- Sensor energy balance: `h(Tm - Tth) = εσ(Tth⁴ - Tsur⁴)`
- Measurement error: `Tbead - T6`

## Analysed operating cases

{}

## Evidence prompts

{}

## Discussion notes

1. **Error ranking:** {}
2. **Shield and airflow effects:** {}
3. **Emissivity and bead size:** {}
4. **Correction and model limitations:** {}

## Instruction video

{}
""".format(
            markdown_table(
                analysed,
                ["Case", "Fan", "Shield", "Air_velocity_m_s", "T7_error_K", "T8_error_K", "T9_error_K", "Maximum_abs_error_K"],
                {"Air_velocity_m_s": "{:.2f}", "T7_error_K": "{:.2f}", "T8_error_K": "{:.2f}", "T9_error_K": "{:.2f}", "Maximum_abs_error_K": "{:.2f}"},
            ),
            "\n".join(f"- {item}" for item in radiation_evidence()),
            st.session_state.rad_interpret_1 or "Not completed",
            st.session_state.rad_interpret_2 or "Not completed",
            st.session_state.rad_interpret_3 or "Not completed",
            st.session_state.rad_interpret_4 or "Not completed",
            VIDEO_URLS[practical],
        )
    return header + model


def safe_file_part(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or fallback


def render_review(practical: str) -> None:
    section_heading(
        "Review and download",
        "Check completion, keep a local copy of raw and processed data, then use your evidence to prepare the report requested in LearnJCU.",
    )
    checks = completion_checks(practical)
    names = SECTIONS[:7]
    rows = []
    for name, complete in zip(names, checks[:7]):
        rows.append({"Stage": name, "Status": "Complete" if complete else "Needs attention"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    complete_count = sum(checks[:7])
    st.progress(complete_count / 7.0, text=f"{complete_count} of 7 learning stages complete")
    if complete_count == 7:
        st.markdown('<div class="status-strip success"><b>Workflow complete.</b> Download all files now and verify them before closing the browser.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-strip warning"><b>Some stages need attention.</b> Downloads remain available so you can preserve work in progress.</div>', unsafe_allow_html=True)

    prefix = practical_prefix(practical)
    student_part = safe_file_part(st.session_state.student_id, "student")
    if practical == PRACTICAL_1:
        raw = st.session_state.conduction_data.copy()
        analysed = analyse_conduction(raw, st.session_state.diameter_mm, st.session_state.heat_fraction)
        if not analysed.empty:
            analysed["Reference_k_W_mK"] = analysed["Material"].map(
                {"Brass": st.session_state.brass_reference_k, "Aluminium": st.session_state.aluminium_reference_k}
            )
            analysed["Deviation_from_reference_pct"] = 100.0 * (
                analysed["Thermal_conductivity_W_mK"] - analysed["Reference_k_W_mK"]
            ) / analysed["Reference_k_W_mK"]
        prac_part = "Prac1_Conduction"
    else:
        raw = st.session_state.radiation_data.copy()
        analysed = analyse_radiation(raw)
        prac_part = "Prac2_Radiation"
    summary = build_summary_markdown(practical)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download raw data CSV",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name=f"ME3512_{prac_part}_{student_part}_raw.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download analysis CSV",
            data=analysed.to_csv(index=False).encode("utf-8"),
            file_name=f"ME3512_{prac_part}_{student_part}_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Download practical record",
            data=summary.encode("utf-8"),
            file_name=f"ME3512_{prac_part}_{student_part}_record.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.subheader("Report-writing check")
    info_cards(
        [
            ("Results", "Show processed evidence", "Use tables, graphs, units and at least one transparent sample calculation."),
            ("Discussion", "Explain, do not narrate", "Connect trends to the governing heat-transfer model and quantify disagreement."),
            ("Limitations", "Separate random and systematic", "Include sensor uncertainty, steady state, geometry and model assumptions."),
        ]
    )
    st.markdown(
        '<div class="status-strip info"><b>Data handling:</b> the app does not intentionally write student entries to a database. Streamlit session data are temporary and can be lost when the browser closes or the app restarts.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Preview practical record"):
        st.markdown(summary)


def main() -> None:
    practical, section = render_sidebar()
    render_header(practical)
    if section == "Prepare":
        render_prepare(practical)
    elif section == "Predict":
        render_predict(practical)
    elif section == "Apparatus and procedure":
        render_apparatus(practical)
    elif section == "Record data":
        render_record(practical)
    elif section == "Calculate and visualise":
        render_calculate(practical)
    elif section == "Explore the concept":
        render_explore(practical)
    elif section == "Interpret results":
        render_interpret(practical)
    else:
        render_review(practical)


if __name__ == "__main__":
    main()
