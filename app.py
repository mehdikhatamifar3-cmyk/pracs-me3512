from __future__ import annotations

import base64
import io
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import streamlit as st

from report_builder import build_practical_report

from thermal_lab_core import (
    COLD_INTERFACE_M,
    CONDUCTION_COLUMNS,
    HOT_INTERFACE_M,
    RADIATION_COLUMNS,
    THERMOCOUPLE_POSITIONS_M,
    analyse_conduction,
    analyse_radiation,
    assigned_online_conduction_data,
    assigned_online_radiation_data,
    blank_conduction_data,
    blank_radiation_data,
    conduction_uncertainty_components,
    conduction_uncertainty_percent,
    demonstration_conduction_data,
    demonstration_radiation_data,
    equilibrium_sensor_temperature_C,
    forced_convection_h,
    linearised_radiation_coefficient_W_m2K,
    normalise_conduction_data,
    normalise_radiation_data,
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
ASSET_DIR = APP_DIR / "assets"
APP_VERSION = "1.4"
JCU_ID_PATTERN = re.compile(r"^\d{8}$")


def load_asset_b64(filename: str) -> str:
    path = ASSET_DIR / filename
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""


JCU_LOGO_B64 = load_asset_b64("jcu_logo.jpg")

CONDUCTION_COLOURS = {
    "hot_contact": "#C2410C",
    "sample": "#B28A00",
    "cold_contact": "#0F766E",
    "measured": "#0B4F8A",
}

RADIATION_COLOURS = {
    "air": "#0B4F8A",
    "bead": "#C2410C",
    "wall": "#B42318",
    "radiation": "#D97706",
    "convection": "#0F766E",
    "shield": "#64748B",
}

PHYSICAL_MODE = "Physical laboratory"
ONLINE_MODE = "Online simulated practical"

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

PERSISTENT_WIDGET_KEYS = {
    "practical",
    "student_name",
    "student_id",
    "group",
    "lab_date",
    "pathway",
    "section_cond",
    "section_rad",
    "cond_video_ready",
    "rad_video_ready",
    "diameter_mm",
    "heat_fraction",
    "brass_reference_k",
    "aluminium_reference_k",
    "unc_v",
    "unc_i",
    "unc_t",
    "unc_d",
    "unc_l",
    "cond_explore_answer",
    "cond_explore_ack",
    "rad_explore_answer",
    "rad_explore_ack",
    "cond_sandbox_q_flux",
    "cond_sandbox_r_hot",
    "cond_sandbox_r_cold",
    "cond_sandbox_k",
    "cond_sandbox_length",
    "rad_sandbox_air",
    "rad_sandbox_wall",
    "rad_sandbox_h",
    "rad_sandbox_emissivity",
    "rad_natural_h",
    "rad_h_scale",
    *(f"cond_quiz_{index}" for index in range(1, 5)),
    *(f"rad_quiz_{index}" for index in range(1, 5)),
    *(f"cond_safety_{index}" for index in range(1, 6)),
    *(f"rad_safety_{index}" for index in range(1, 6)),
    *(f"cond_interpret_{index}" for index in range(1, 5)),
    *(f"rad_interpret_{index}" for index in range(1, 5)),
}


def preserve_widget_values_across_pages() -> None:
    """Interrupt Streamlit cleanup for widgets hidden on another section."""
    cache = st.session_state.get("_persistent_widget_values", {})
    if not isinstance(cache, dict):
        cache = {}
    for key in PERSISTENT_WIDGET_KEYS:
        if key in st.session_state:
            value = st.session_state[key]
            st.session_state[key] = value
            cache[key] = value
    for key, value in cache.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["_persistent_widget_values"] = cache


def initialise_state() -> None:
    defaults = {
        "student_name": "",
        "student_id": "",
        "group": "",
        "lab_date": date.today(),
        "pathway": PHYSICAL_MODE,
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
        "unc_v": 0.01,
        "unc_i": 0.01,
        "unc_t": 0.10,
        "unc_d": 0.10,
        "unc_l": 0.10,
        "cond_sandbox_q_flux": 25.0,
        "cond_sandbox_r_hot": 2.0,
        "cond_sandbox_r_cold": 8.0,
        "cond_sandbox_k": 120,
        "cond_sandbox_length": 30,
        "rad_sandbox_air": 25.0,
        "rad_sandbox_wall": 120.0,
        "rad_sandbox_h": 25.0,
        "rad_sandbox_emissivity": 0.98,
        "rad_natural_h": 70.0,
        "rad_h_scale": 1.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "_conduction_data_store" not in st.session_state:
        st.session_state["_conduction_data_store"] = normalise_conduction_data(
            st.session_state.conduction_data
        ).copy(deep=True)
    if "_radiation_data_store" not in st.session_state:
        st.session_state["_radiation_data_store"] = normalise_radiation_data(
            st.session_state.radiation_data
        ).copy(deep=True)
    st.session_state.conduction_data = normalise_conduction_data(
        st.session_state["_conduction_data_store"]
    ).copy(deep=True)
    st.session_state.radiation_data = normalise_radiation_data(
        st.session_state["_radiation_data_store"]
    ).copy(deep=True)


preserve_widget_values_across_pages()
initialise_state()
if st.session_state.pathway == "Demonstration data exploration":
    st.session_state.pathway = ONLINE_MODE


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
      .header-mark { min-width:112px; display:flex; justify-content:flex-end; }
      .header-logo { background:#FFFFFF; border-radius:12px; padding:7px; box-shadow:0 4px 14px rgba(1,31,54,.16); }
      .header-logo img { display:block; width:96px; max-width:100%; height:auto; }
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
        .header-mark { justify-content:flex-start; }
        .header-logo img { width:88px; }
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


def online_mode_active() -> bool:
    return st.session_state.pathway == ONLINE_MODE


def active_conduction_data() -> pd.DataFrame:
    if online_mode_active():
        return assigned_online_conduction_data(st.session_state.student_id)
    stored = st.session_state.get("_conduction_data_store")
    if stored is None:
        stored = st.session_state.get("conduction_data", blank_conduction_data())
    return normalise_conduction_data(stored).copy(deep=True)


def active_radiation_data() -> pd.DataFrame:
    if online_mode_active():
        return assigned_online_radiation_data(st.session_state.student_id)
    stored = st.session_state.get("_radiation_data_store")
    if stored is None:
        stored = st.session_state.get("radiation_data", blank_radiation_data())
    return normalise_radiation_data(stored).copy(deep=True)


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
    logo_html = (
        f'<img src="data:image/jpeg;base64,{JCU_LOGO_B64}" alt="James Cook University logo">'
        if JCU_LOGO_B64
        else '<div style="font-size:24px;font-weight:850;color:#0B4F8A">JCU</div>'
    )
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="eyebrow">ME3512 · Heat and Mass Transfer · ThermalLab</div>
            <h1>{html_escape(PRACTICAL_TITLES[practical])}</h1>
            <div class="subtitle">{html_escape(PRACTICAL_SUBTITLES[practical])}</div>
          </div>
          <div class="header-mark">
            <div class="header-logo">{logo_html}</div>
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
        valid_data = len(valid_conduction_rows(active_conduction_data())) >= 4
        safety = all(st.session_state.cond_safety.values()) and len(st.session_state.cond_safety) == 5
        interpretations = all(st.session_state.get(f"cond_interpret_{i}", "").strip() for i in range(1, 5))
    else:
        valid_data = len(valid_radiation_rows(active_radiation_data())) >= 4
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
    fig, ax = plt.subplots(figsize=(9.4, 8.3))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 15.2)
    ax.axis("off")

    # Vertical test duct and inlet transition.
    duct_left, duct_right = 4.6, 7.6
    ax.add_patch(
        Rectangle(
            (duct_left, 2.6),
            duct_right - duct_left,
            11.1,
            facecolor="#EEF5F8",
            edgecolor="#405465",
            linewidth=2.0,
        )
    )
    ax.add_patch(Rectangle((4.25, 2.25), 3.7, 0.45, facecolor="#A7B8C5", edgecolor="#405465", linewidth=1.4))

    # Blower and throttle plate at the inlet.
    ax.add_patch(
        FancyBboxPatch(
            (0.9, 0.65),
            3.75,
            2.0,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor="#61788C",
            edgecolor="#334155",
            linewidth=1.6,
        )
    )
    ax.add_patch(Circle((2.0, 1.65), 0.55, facecolor="#405465", edgecolor="#D9E5ED", linewidth=1.3))
    ax.add_patch(Circle((2.0, 1.65), 0.12, facecolor="#E53935", edgecolor="white", linewidth=0.9))
    ax.text(2.9, 1.65, "VARIABLE-SPEED\nBLOWER", ha="center", va="center", fontsize=8.3, color="white", fontweight="bold")
    ax.add_patch(Rectangle((4.65, 0.75), 2.95, 1.7, facecolor="#9FB2C1", edgecolor="#405465", linewidth=1.5))
    for y in np.linspace(0.92, 2.28, 7):
        ax.plot([4.85, 7.38], [y, y], color="#52697D", lw=1.0)
    ax.plot([6.95, 7.95], [1.55, 1.15], color="#334155", lw=2.0)
    ax.add_patch(Circle((7.95, 1.15), 0.11, facecolor="#F2A900", edgecolor="#334155", linewidth=0.8))
    ax.annotate(
        "Throttle plate\nsets the air speed",
        xy=(7.55, 1.35),
        xytext=(9.45, 1.05),
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
        arrowprops={"arrowstyle": "->", "color": "#64748B", "lw": 1.1},
    )

    # Airflow through the duct.
    for x in (5.45, 6.75):
        for y in (3.4, 5.6, 7.6, 9.5, 12.1):
            ax.add_patch(
                FancyArrowPatch(
                    (x, y - 0.45),
                    (x, y + 0.45),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color=RADIATION_COLOURS["air"],
                    lw=1.55,
                    alpha=0.88,
                )
            )
    ax.text(6.1, 13.35, "AIRFLOW", ha="center", va="center", fontsize=9.5, color=RADIATION_COLOURS["air"], fontweight="bold")

    # T6 is upstream of the heated measurement section.
    ax.plot([10.4, 7.25, 6.35], [4.75, 4.75, 4.75], color="#4C7F94", lw=2.0)
    ax.plot([6.35, 6.15], [4.75, 4.15], color="#4C7F94", lw=2.0)
    ax.add_patch(Circle((6.15, 4.08), 0.08, facecolor=RADIATION_COLOURS["air"], edgecolor="white", linewidth=0.8, zorder=5))
    ax.text(10.55, 4.75, "T6  reference air temperature", ha="left", va="center", fontsize=9.5, color=RADIATION_COLOURS["air"], fontweight="bold")

    # Anemometer in the lower-middle duct.
    ax.plot([10.2, 7.15, 6.35], [6.45, 6.45, 6.45], color="#334155", lw=1.7)
    ax.add_patch(Rectangle((7.62, 6.15), 0.72, 0.60, facecolor="#334155", edgecolor="#172033", linewidth=1.0))
    ax.add_patch(Circle((6.20, 6.45), 0.32, facecolor="#E5EDF2", edgecolor="#334155", linewidth=1.2))
    for angle in (0, 120, 240):
        theta = math.radians(angle)
        ax.plot([6.20, 6.20 + 0.27 * math.cos(theta)], [6.45, 6.45 + 0.27 * math.sin(theta)], color="#334155", lw=1.2)
    ax.text(10.35, 6.45, "Anemometer  V", ha="left", va="center", fontsize=9.5, color="#334155", fontweight="bold")

    # Heated wall section and coils.
    heater_y0, heater_y1 = 8.0, 11.6
    for x, direction in ((4.22, -1), (7.98, 1)):
        ax.add_patch(Rectangle((x - 0.16, heater_y0), 0.32, heater_y1 - heater_y0, facecolor="#FDE5E3", edgecolor="#B42318", linewidth=1.2))
        ys = np.linspace(heater_y0 + 0.18, heater_y1 - 0.18, 18)
        xs = x + 0.11 * np.where(np.arange(len(ys)) % 2 == 0, -1, 1)
        ax.plot(xs, ys, color="#D64545", lw=1.4)
    ax.annotate(
        "Electrical heaters\ncreate hot radiative walls",
        xy=(7.98, 9.3),
        xytext=(10.2, 8.8),
        ha="left",
        va="center",
        fontsize=9.3,
        color=RADIATION_COLOURS["wall"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RADIATION_COLOURS["wall"], "lw": 1.1},
    )

    # T10 monitors the heated wall near the test beads.
    ax.plot([3.3, 4.45], [12.55, 12.55], color=RADIATION_COLOURS["wall"], lw=1.8)
    ax.add_patch(Circle((4.48, 12.55), 0.08, facecolor=RADIATION_COLOURS["wall"], edgecolor="white", linewidth=0.8, zorder=5))
    ax.annotate(
        "T10  heated-wall temperature",
        xy=(4.48, 12.55),
        xytext=(0.55, 12.75),
        ha="left",
        va="center",
        fontsize=9.5,
        color=RADIATION_COLOURS["wall"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RADIATION_COLOURS["wall"], "lw": 1.1},
    )

    # Three sensing beads with separate, directly labelled probes.
    probe_rows = [
        (11.25, 6.70, 0.09, "#D7DDE3", "T7  polished 0.5 mm  (ε ≈ 0.17)"),
        (10.75, 6.35, 0.09, "#111827", "T8  black 0.5 mm  (ε ≈ 0.98)"),
        (10.25, 5.95, 0.22, "#111827", "T9  black 3 mm  (ε ≈ 0.98)"),
    ]
    for y, bead_x, radius, colour, label in probe_rows:
        ax.plot([10.55, 7.45, bead_x + radius], [y, y, y], color="#5D7283", lw=1.6)
        ax.add_patch(Circle((bead_x, y), radius, facecolor=colour, edgecolor="#111827", linewidth=1.1, zorder=6))
        ax.text(10.72, y, label, ha="left", va="center", fontsize=9.1, color="#334155")

    # Movable shield around the bead zone, shown as a dashed sleeve.
    ax.add_patch(
        Rectangle(
            (5.35, 9.65),
            1.85,
            2.15,
            fill=False,
            edgecolor=RADIATION_COLOURS["shield"],
            linewidth=2.1,
            linestyle=(0, (5, 3)),
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (5.05, 9.75),
            (5.05, 11.70),
            arrowstyle="<->",
            mutation_scale=14,
            color=RADIATION_COLOURS["shield"],
            lw=1.5,
        )
    )
    ax.annotate(
        "Movable radiation shield\nDOWN: beads exposed to hot wall\nUP: hot wall view blocked",
        xy=(5.35, 10.65),
        xytext=(0.55, 9.65),
        ha="left",
        va="center",
        fontsize=9.1,
        color=RADIATION_COLOURS["shield"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RADIATION_COLOURS["shield"], "lw": 1.1},
    )

    ax.text(
        0.55,
        14.65,
        "HT16C radiation-error apparatus",
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#183A57",
    )
    ax.text(
        0.55,
        14.15,
        "Follow the airflow from the blower to T6, then to the heated test section and the three beads.",
        ha="left",
        va="center",
        fontsize=9.3,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)
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
            [PHYSICAL_MODE, ONLINE_MODE],
            key="pathway",
            help="Online students receive a stable assigned dataset automatically from their JCU ID.",
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
            if online_mode_active()
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


def store_conduction_data(data: pd.DataFrame) -> None:
    normalised = normalise_conduction_data(data).copy(deep=True)
    st.session_state.conduction_data = normalised.copy(deep=True)
    st.session_state["_conduction_data_store"] = normalised.copy(deep=True)


def store_radiation_data(data: pd.DataFrame) -> None:
    normalised = normalise_radiation_data(data).copy(deep=True)
    st.session_state.radiation_data = normalised.copy(deep=True)
    st.session_state["_radiation_data_store"] = normalised.copy(deep=True)


def replace_conduction_data(data: pd.DataFrame) -> None:
    store_conduction_data(data)
    st.session_state.cond_editor_version += 1


def replace_radiation_data(data: pd.DataFrame) -> None:
    store_radiation_data(data)
    st.session_state.rad_editor_version += 1


def render_conduction_record() -> None:
    st.markdown(
        '<div class="concept-card"><div class="label">Record first, calculate second</div><div class="big">Preserve all eight temperatures as one operating point.</div><div class="text">T1-T8, voltage and current must belong to the same stable condition. Do not mix readings taken before and after a control change.</div></div>',
        unsafe_allow_html=True,
    )
    column_config = {
        "Material": st.column_config.SelectboxColumn("Material", options=["Brass", "Aluminium", "Other"], required=True),
        "Trial": st.column_config.TextColumn("Trial"),
        "Voltage_V": st.column_config.NumberColumn("V (V)", min_value=0.0, format="%.3f"),
        "Current_A": st.column_config.NumberColumn("I (A)", min_value=0.0, format="%.3f"),
        "Water_flow_L_min": st.column_config.NumberColumn("Fw (L/min)", min_value=0.0, format="%.3f"),
    }
    for index in range(1, 9):
        column_config[f"T{index}_C"] = st.column_config.NumberColumn(f"T{index} (°C)", format="%.2f")
    if online_mode_active():
        assigned = active_conduction_data()
        if identity_complete():
            st.markdown(
                '<div class="status-strip success"><b>Your online dataset has been assigned automatically.</b> It is linked deterministically to your JCU ID and is read-only. Use these values for the remaining analysis stages.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-strip warning"><b>Enter your name and 8-digit JCU ID in Prepare.</b> A temporary dataset is shown now; your final assigned values are locked by your JCU ID.</div>',
                unsafe_allow_html=True,
            )
        st.dataframe(
            assigned,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=330,
        )
        valid = valid_conduction_rows(assigned)
        incomplete = 0
    else:
        b1, b2, spacer = st.columns([1, 1, 2.2])
        with b1:
            if st.button("Load example data", use_container_width=True, help="For instructor testing only; do not present these values as your measurements."):
                replace_conduction_data(demonstration_conduction_data())
                st.rerun()
        with b2:
            if st.button("Reset to blank table", use_container_width=True):
                replace_conduction_data(blank_conduction_data())
                st.rerun()
        st.caption("Enter only your group's measured values. The table is retained when you move between workflow sections.")
        physical_data = active_conduction_data()
        edited = st.data_editor(
            physical_data,
            key=f"cond_editor_{st.session_state.cond_editor_version}",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=330,
        )
        store_conduction_data(edited)
        valid = valid_conduction_rows(active_conduction_data())
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
    if online_mode_active():
        assigned = active_radiation_data()
        if identity_complete():
            st.markdown(
                '<div class="status-strip success"><b>Your online dataset has been assigned automatically.</b> It is linked deterministically to your JCU ID and is read-only. Use these four cases for the remaining analysis stages.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-strip warning"><b>Enter your name and 8-digit JCU ID in Prepare.</b> A temporary dataset is shown now; your final assigned values are locked by your JCU ID.</div>',
                unsafe_allow_html=True,
            )
        st.dataframe(
            assigned,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=300,
        )
        valid = valid_radiation_rows(assigned)
        incomplete = 0
    else:
        b1, b2, spacer = st.columns([1, 1, 2.2])
        with b1:
            if st.button("Load example data", use_container_width=True, key="load_rad_demo", help="For instructor testing only; do not present these values as your measurements."):
                replace_radiation_data(demonstration_radiation_data())
                st.rerun()
        with b2:
            if st.button("Reset to blank table", use_container_width=True, key="reset_rad"):
                replace_radiation_data(blank_radiation_data())
                st.rerun()
        st.caption("Enter only your group's measured values. The table is retained when you move between workflow sections.")
        physical_data = active_radiation_data()
        edited = st.data_editor(
            physical_data,
            key=f"rad_editor_{st.session_state.rad_editor_version}",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            height=300,
        )
        store_radiation_data(edited)
        valid = valid_radiation_rows(active_radiation_data())
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
        "Enter physical measurements or use the read-only dataset assigned automatically for the online pathway.",
    )
    if practical == PRACTICAL_1:
        render_conduction_record()
    else:
        render_radiation_record()


def conduction_temperature_plot(raw_row: pd.Series, result: pd.Series) -> plt.Figure:
    temperatures = np.array([float(raw_row[f"T{i}_C"]) for i in range(1, 9)])
    fig = plt.figure(figsize=(11.4, 7.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.15], hspace=0.23)
    ax = fig.add_subplot(grid[0])
    formula_ax = fig.add_subplot(grid[1])
    ax.scatter(
        THERMOCOUPLE_POSITIONS_M * 1000,
        temperatures,
        s=68,
        color="#0B4F8A",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="Measured thermocouples",
    )
    regions = [
        (np.linspace(0.000, HOT_INTERFACE_M, 60), result["Hot_slope_K_m"], result["Hot_intercept_C"], "Hot bar", "#C2410C"),
        (np.linspace(HOT_INTERFACE_M, COLD_INTERFACE_M, 60), result["Sample_slope_K_m"], result["Sample_intercept_C"], "Sample", "#B28A00"),
        (np.linspace(COLD_INTERFACE_M, 0.105, 60), result["Cold_slope_K_m"], result["Cold_intercept_C"], "Cold bar", "#0F766E"),
    ]
    formula_items = []
    r_squared_values = [result["Hot_fit_R2"], result["Sample_fit_R2"], result["Cold_fit_R2"]]
    for (x, slope, intercept, label, colour), r_squared in zip(regions, r_squared_values):
        ax.plot(x * 1000, slope * x + intercept, lw=2.4, color=colour, label=f"{label} fitted line")
        slope_per_mm = float(slope) / 1000.0
        intercept_sign = "+" if float(intercept) >= 0 else "−"
        formula_items.append(
            (
                label,
                f"T = {slope_per_mm:.4f}x {intercept_sign} {abs(float(intercept)):.2f}",
                f"R² = {float(r_squared):.4f}",
                colour,
            )
        )

    interface_points = [
        (HOT_INTERFACE_M, result["Hot_bar_face_C"], result["Sample_hot_face_C"], "Hot contact"),
        (COLD_INTERFACE_M, result["Sample_cold_face_C"], result["Cold_bar_face_C"], "Cold contact"),
    ]
    all_temperatures = np.concatenate(
        [
            temperatures,
            np.array(
                [
                    result["Hot_bar_face_C"],
                    result["Sample_hot_face_C"],
                    result["Sample_cold_face_C"],
                    result["Cold_bar_face_C"],
                ],
                dtype=float,
            ),
        ]
    )
    temperature_span = max(float(np.ptp(all_temperatures)), 1.0)
    ax.set_ylim(float(np.min(all_temperatures)) - 0.16 * temperature_span, float(np.max(all_temperatures)) + 0.17 * temperature_span)
    ax.set_xlim(-4.0, 109.0)

    for x, label in [(HOT_INTERFACE_M, "hot interface"), (COLD_INTERFACE_M, "cold interface")]:
        ax.axvline(x * 1000, color="#94A3B8", ls="--", lw=1.2)
        ax.annotate(
            label,
            xy=(x * 1000, ax.get_ylim()[1]),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#64748B",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )

    for point_index, (x_m, left_temperature, right_temperature, contact_name) in enumerate(interface_points):
        x_mm = x_m * 1000.0
        ax.plot(
            [x_mm, x_mm],
            [left_temperature, right_temperature],
            color="#B42318",
            lw=4,
            solid_capstyle="round",
            label="Contact temperature jump" if point_index == 0 else None,
            zorder=4,
        )
        ax.scatter(
            [x_mm, x_mm],
            [left_temperature, right_temperature],
            s=58,
            facecolor="white",
            edgecolor="#B42318",
            linewidth=1.8,
            zorder=6,
        )
        label_offset = 9 if point_index == 0 else -9
        label_alignment = "left" if point_index == 0 else "right"
        left_vertical_offset = 9 if point_index == 0 else -6
        left_vertical_alignment = "bottom" if point_index == 0 else "top"
        ax.annotate(
            f"L  {left_temperature:.2f} °C",
            xy=(x_mm, left_temperature),
            xytext=(label_offset, left_vertical_offset),
            textcoords="offset points",
            ha=label_alignment,
            va=left_vertical_alignment,
            fontsize=8.1,
            fontweight="bold",
            color="#7F1D1D",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFF7F6", "edgecolor": "#F2C6C2"},
        )
        ax.annotate(
            f"R  {right_temperature:.2f} °C",
            xy=(x_mm, right_temperature),
            xytext=(label_offset, -9),
            textcoords="offset points",
            ha=label_alignment,
            va="top",
            fontsize=8.1,
            fontweight="bold",
            color="#7F1D1D",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFF7F6", "edgecolor": "#F2C6C2"},
        )
        delta_x = x_mm + (5.0 if point_index == 0 else -5.0)
        ax.text(
            delta_x,
            0.5 * (float(left_temperature) + float(right_temperature)),
            f"ΔT = {left_temperature-right_temperature:.2f} K",
            ha="left" if point_index == 0 else "right",
            va="center",
            fontsize=8.1,
            color="#7F1D1D",
            fontweight="bold",
        )
    for index, (x, y) in enumerate(zip(THERMOCOUPLE_POSITIONS_M * 1000, temperatures), start=1):
        offset = 10 if index % 2 else 7
        ax.annotate(f"T{index}", (x, y), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color="#334155")
    ax.set_xlabel("Position x (mm)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature versus distance", loc="left", pad=33, fontweight="bold", color="#183A57")
    ax.grid(alpha=0.20)
    ax.legend(
        ncol=3,
        frameon=False,
        fontsize=8.1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.095),
        columnspacing=1.4,
        handlelength=2.2,
    )

    formula_ax.set_xlim(0, 1)
    formula_ax.set_ylim(0, 1)
    formula_ax.axis("off")
    formula_ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.08),
            0.98,
            0.84,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#F8FAFC",
            edgecolor="#D7E0E9",
            linewidth=1.0,
        )
    )
    formula_ax.text(0.03, 0.77, "Fitted lines (T in °C; x in mm)", fontsize=9.1, fontweight="bold", color="#334155", va="center")
    for x_position, (label, formula, r_squared, colour) in zip((0.03, 0.355, 0.68), formula_items):
        formula_ax.plot([x_position, x_position + 0.055], [0.53, 0.53], color=colour, lw=3.0, solid_capstyle="round")
        formula_ax.text(x_position + 0.065, 0.56, label, fontsize=8.7, fontweight="bold", color="#334155", va="center")
        formula_ax.text(x_position + 0.065, 0.34, formula, fontsize=8.5, family="monospace", color="#334155", va="center")
        formula_ax.text(x_position + 0.065, 0.17, r_squared, fontsize=8.1, color="#64748B", va="center")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.04)
    return fig


def conduction_fit_table(result: pd.Series) -> pd.DataFrame:
    rows = []
    for region, slope_key, intercept_key, r_squared_key, sensors in [
        ("Hot bar", "Hot_slope_K_m", "Hot_intercept_C", "Hot_fit_R2", "T1-T3"),
        ("Sample", "Sample_slope_K_m", "Sample_intercept_C", "Sample_fit_R2", "T4-T5"),
        ("Cold bar", "Cold_slope_K_m", "Cold_intercept_C", "Cold_fit_R2", "T6-T8"),
    ]:
        slope_per_mm = float(result[slope_key]) / 1000.0
        intercept = float(result[intercept_key])
        sign = "+" if intercept >= 0 else "−"
        rows.append(
            {
                "Region": region,
                "Sensors": sensors,
                "Fitted_formula_x_mm": f"T = {slope_per_mm:.4f}x {sign} {abs(intercept):.2f}",
                "R_squared": float(result[r_squared_key]),
            }
        )
    return pd.DataFrame(rows)


def conduction_contact_temperature_table(result: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Interface": "Hot contact",
                "Position_mm": HOT_INTERFACE_M * 1000.0,
                "Left_side": "Hot bar",
                "T_left_C": float(result["Hot_bar_face_C"]),
                "Right_side": "Sample",
                "T_right_C": float(result["Sample_hot_face_C"]),
                "Temperature_jump_K": float(result["Hot_contact_jump_K"]),
            },
            {
                "Interface": "Cold contact",
                "Position_mm": COLD_INTERFACE_M * 1000.0,
                "Left_side": "Sample",
                "T_left_C": float(result["Sample_cold_face_C"]),
                "Right_side": "Cold bar",
                "T_right_C": float(result["Cold_bar_face_C"]),
                "Temperature_jump_K": float(result["Cold_contact_jump_K"]),
            },
        ]
    )


def conduction_resistance_plot(result: pd.Series) -> plt.Figure:
    labels = ["Hot contact", "Sample", "Cold contact"]
    drops = np.array([
        result["Hot_contact_jump_K"],
        result["Sample_drop_K"],
        result["Cold_contact_jump_K"],
    ])
    colours = [
        CONDUCTION_COLOURS["hot_contact"],
        CONDUCTION_COLOURS["sample"],
        CONDUCTION_COLOURS["cold_contact"],
    ]
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


def conduction_parameter_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("V", "Heater voltage", "V"),
            ("I", "Heater current", "A"),
            ("fQ", "Fraction of electrical power assumed to enter one-dimensional conduction", "-"),
            ("Q", "Heat rate through the bar: Q = fQ V I", "W"),
            ("D", "Bar and contact diameter", "m"),
            ("A", "Cross-sectional area: A = pi D^2 / 4", "m^2"),
            ("q''", "Heat flux: q'' = Q/A", "W/m^2"),
            ("m = dT/dx", "Slope of a fitted temperature line", "K/m"),
            ("k", "Sample thermal conductivity: k = -q''/m_sample", "W/(m K)"),
            ("Delta Tc", "Extrapolated left-side minus right-side interface temperature", "K"),
            ("R''c", "Area-specific contact resistance: R''c = Delta Tc/q''", "m^2 K/W"),
        ],
        columns=["Symbol", "Definition", "Unit"],
    )


def radiation_parameter_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Tm", "Actual medium or air temperature required", "K or °C outside fourth powers"),
            ("Tth", "Thermocouple-bead temperature indicated by the sensor", "K"),
            ("Tsur", "Effective temperature of the surrounding surfaces seen by the bead", "K"),
            ("h", "Convective heat-transfer coefficient between air and bead", "W/(m^2 K)"),
            ("epsilon", "Total hemispherical emissivity of the bead surface", "-"),
            ("sigma", "Stefan-Boltzmann constant, 5.670374419e-8", "W/(m^2 K^4)"),
            ("D", "Thermocouple bead diameter used in Re, Nu and h", "m"),
            ("ReD", "Reynolds number based on bead diameter", "-"),
            ("NuD", "Nusselt number used to estimate h", "-"),
        ],
        columns=["Symbol", "Definition", "Unit"],
    )


def render_conduction_calculate() -> None:
    valid = valid_conduction_rows(active_conduction_data())
    if valid.empty:
        st.warning("Enter at least one complete operating point in Record data before calculating.")
        return
    with st.expander("Governing formulas and parameter definitions", expanded=True):
        formula_column, definition_column = st.columns([0.86, 1.14])
        with formula_column:
            st.latex(r"Q=f_QVI,\qquad A=\frac{\pi D^2}{4},\qquad q''=\frac{Q}{A}")
            st.latex(r"m_s=\frac{dT}{dx},\qquad k=-\frac{q''}{m_s}")
            st.latex(r"T_{face}=mx_{interface}+b")
            st.latex(r"\Delta T_c=T_{left}-T_{right},\qquad R''_c=\frac{\Delta T_c}{q''}")
            st.caption("The hot bar, sample and cold bar are fitted separately. Temperatures on the two sides of an interface come from the two relevant fitted lines at the same x-position.")
        with definition_column:
            st.dataframe(conduction_parameter_definitions(), use_container_width=True, hide_index=True)

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

    st.subheader("Fitted lines and extrapolated contact temperatures")
    fit_column, contact_column = st.columns([0.92, 1.45])
    with fit_column:
        st.caption("Each equation uses T in °C and distance x in mm.")
        st.dataframe(
            conduction_fit_table(result),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Fitted_formula_x_mm": st.column_config.TextColumn("Fitted line (x in mm)", width="large"),
                "R_squared": st.column_config.NumberColumn("R²", format="%.4f"),
            },
        )
    with contact_column:
        st.caption("Left and right are defined while moving from the heater towards the cooling-water end.")
        st.dataframe(
            conduction_contact_temperature_table(result),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Position_mm": st.column_config.NumberColumn("x (mm)", format="%.1f"),
                "T_left_C": st.column_config.NumberColumn("T left (°C)", format="%.2f"),
                "T_right_C": st.column_config.NumberColumn("T right (°C)", format="%.2f"),
                "Temperature_jump_K": st.column_config.NumberColumn("Jump (K)", format="%.2f"),
            },
        )
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
        sample_delta_x = 0.015
        sample_delta_t = float(raw_row["T5_C"] - raw_row["T4_C"])
        st.latex(r"A=\frac{\pi D^2}{4}")
        st.write(f"A = π({st.session_state.diameter_mm/1000:.4f} m)²/4 = {area:.6e} m²")
        st.latex(r"Q=f_QVI,\qquad q''=\frac{Q}{A}")
        st.write(
            f"Q = {st.session_state.heat_fraction:.2f} × {result['Voltage_V']:.3f} × {result['Current_A']:.3f} "
            f"= {result['Assumed_conduction_heat_W']:.3f} W; q'' = {result['Heat_flux_W_m2']:.1f} W/m²"
        )
        st.latex(r"m_s=\frac{T_5-T_4}{x_5-x_4},\qquad k=-\frac{q''}{m_s}")
        st.write(
            f"m_s = ({raw_row['T5_C']:.2f} - {raw_row['T4_C']:.2f}) K / {sample_delta_x:.3f} m "
            f"= {sample_delta_t/sample_delta_x:.2f} K/m. Therefore k = -{result['Heat_flux_W_m2']:.1f} / "
            f"({result['Sample_slope_K_m']:.2f}) = {result['Thermal_conductivity_W_mK']:.2f} W/(m·K)."
        )
        st.latex(r"T_{face}=mx_{interface}+b")
        st.write(
            f"At the hot interface x = {HOT_INTERFACE_M:.4f} m: Tleft = {result['Hot_slope_K_m']:.2f}({HOT_INTERFACE_M:.4f}) "
            f"+ {result['Hot_intercept_C']:.2f} = {result['Hot_bar_face_C']:.2f} °C, while Tright = "
            f"{result['Sample_slope_K_m']:.2f}({HOT_INTERFACE_M:.4f}) + {result['Sample_intercept_C']:.2f} = "
            f"{result['Sample_hot_face_C']:.2f} °C."
        )
        st.latex(r"R''_c=\frac{\Delta T_{interface}}{q''}")
        st.write(
            f"Hot contact: {result['Hot_contact_jump_K']:.2f} K / {result['Heat_flux_W_m2']:.1f} W/m² = {result['Hot_contact_Rpp_m2K_W']:.3e} m²·K/W."
        )
        st.write(
            f"Cold contact: {result['Cold_contact_jump_K']:.2f} K / {result['Heat_flux_W_m2']:.1f} W/m² = {result['Cold_contact_Rpp_m2K_W']:.3e} m²·K/W."
        )

    with st.expander("Quantitative uncertainty estimate for k"):
        st.markdown("**Propagation model used by the app**")
        st.latex(r"k=\frac{4f_QVIL}{\pi D^2\Delta T},\qquad \Delta T=T_4-T_5")
        st.latex(
            r"\left(\frac{u_k}{k}\right)^2="
            r"\left(\frac{u_V}{V}\right)^2+\left(\frac{u_I}{I}\right)^2+"
            r"\left(\frac{u_L}{L}\right)^2+\left(2\frac{u_D}{D}\right)^2+"
            r"\left(\frac{u_{\Delta T}}{\Delta T}\right)^2"
        )
        st.latex(r"u_{\Delta T}=\sqrt{u_{T4}^2+u_{T5}^2}=\sqrt{2}\,u_T")
        st.caption("Independent input uncertainties are combined by root-sum-of-squares (RSS). The heat fraction fQ is treated as an explicit model assumption, not an instrument uncertainty.")
        c1, c2, c3, c4, c5 = st.columns(5)
        d_v = c1.number_input("±V (V)", min_value=0.0, max_value=5.0, step=0.01, key="unc_v")
        d_i = c2.number_input("±I (A)", min_value=0.0, max_value=2.0, step=0.01, key="unc_i")
        d_t = c3.number_input("±T (K)", min_value=0.0, max_value=5.0, step=0.05, key="unc_t")
        d_d = c4.number_input("±D (mm)", min_value=0.0, max_value=5.0, step=0.05, key="unc_d")
        d_l = c5.number_input("±Δx (mm)", min_value=0.0, max_value=5.0, step=0.05, key="unc_l")
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
        uncertainty_components = conduction_uncertainty_components(
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
            component_labels = [
                ("Voltage", "uV/V"),
                ("Current", "uI/I"),
                ("Sensor spacing", "uL/L"),
                ("Diameter", "2uD/D"),
                ("Temperature difference", "uΔT/ΔT"),
            ]
            component_table = pd.DataFrame(
                [
                    {
                        "Input": label,
                        "Relative term": expression,
                        "Relative uncertainty (%)": 100.0 * uncertainty_components[label],
                        "Share of variance (%)": 100.0 * uncertainty_components[label] ** 2 / uncertainty_components["Combined"] ** 2
                        if uncertainty_components["Combined"] > 0
                        else 0.0,
                    }
                    for label, expression in component_labels
                ]
            )
            substitution = " + ".join(
                f"({100.0 * uncertainty_components[label]:.2f}%)²" for label, _ in component_labels
            )
            st.write(f"For this operating point: uₖ/k = √[{substitution}] = **{uncertainty:.2f}%**.")
            st.dataframe(
                component_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Relative uncertainty (%)": st.column_config.NumberColumn("Relative uncertainty (%)", format="%.2f"),
                    "Share of variance (%)": st.column_config.NumberColumn("Share of combined variance (%)", format="%.1f"),
                },
            )
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


def radiation_h_sensitivity_plot(
    row: pd.Series,
    measured_bead_C: float,
    emissivity: float,
    selected_h_W_m2K: float,
) -> plt.Figure:
    """Show how the assumed h changes the radiation-corrected medium temperature."""
    selected_h = max(float(selected_h_W_m2K), 1.0)
    h_min = max(2.0, selected_h / 10.0)
    h_max = max(650.0, selected_h * 2.4)
    h_values = np.geomspace(h_min, h_max, 240)
    surrounding = float(row["T10_wall_C"])
    reference_air = float(row["T6_air_C"])
    corrected = np.array(
        [
            radiation_corrected_medium_temperature_C(
                measured_bead_C,
                surrounding,
                h_value,
                emissivity,
            )
            for h_value in h_values
        ]
    )
    selected_corrected = radiation_corrected_medium_temperature_C(
        measured_bead_C,
        surrounding,
        selected_h,
        emissivity,
    )

    fig, (temperature_ax, correction_ax) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0]},
    )
    temperature_ax.semilogx(
        h_values,
        corrected,
        color=RADIATION_COLOURS["air"],
        lw=2.6,
        label="Corrected medium temperature Tm",
    )
    temperature_ax.axhline(reference_air, color=RADIATION_COLOURS["convection"], lw=1.7, ls="--", label="Independent T6 air reference")
    temperature_ax.axhline(measured_bead_C, color=RADIATION_COLOURS["bead"], lw=1.5, ls=":", label="Measured T8 bead")
    temperature_ax.axvline(selected_h, color="#64748B", lw=1.3, ls="--")
    temperature_ax.scatter([selected_h], [selected_corrected], s=72, color=RADIATION_COLOURS["air"], edgecolor="white", linewidth=1.2, zorder=5)
    temperature_ax.annotate(
        f"selected h = {selected_h:.1f}\nTm = {selected_corrected:.2f} °C",
        xy=(selected_h, selected_corrected),
        xytext=(18, -48),
        textcoords="offset points",
        fontsize=8.6,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D7E0E9"},
        arrowprops={"arrowstyle": "->", "color": "#64748B", "lw": 1.0},
    )
    temperature_ax.set_ylabel("Temperature (°C)")
    temperature_ax.grid(which="both", alpha=0.18)
    temperature_ax.legend(ncol=3, frameon=False, fontsize=8.2, loc="lower center", bbox_to_anchor=(0.5, 1.04))

    correction_magnitude = np.abs(corrected - measured_bead_C)
    correction_ax.semilogx(h_values, correction_magnitude, color=RADIATION_COLOURS["radiation"], lw=2.5)
    correction_ax.axvline(selected_h, color="#64748B", lw=1.3, ls="--")
    correction_ax.scatter(
        [selected_h],
        [abs(selected_corrected - measured_bead_C)],
        s=62,
        color=RADIATION_COLOURS["radiation"],
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    correction_ax.set_xlabel("Convective heat-transfer coefficient h (W/m²·K), logarithmic scale")
    correction_ax.set_ylabel("|Tm - T8| (K)")
    correction_ax.set_title("Radiation correction magnitude decreases approximately as 1/h", loc="left", fontsize=10, fontweight="bold", color="#334155")
    correction_ax.grid(which="both", alpha=0.18)
    for axis in (temperature_ax, correction_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Sensitivity of the energy-balance correction to h",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#183A57",
    )
    fig.tight_layout(h_pad=1.55, rect=[0, 0, 1, 0.91])
    return fig


def render_radiation_calculate() -> None:
    analysed = analyse_radiation(active_radiation_data())
    if analysed.empty:
        st.warning("Enter at least one complete operating case in Record data before calculating.")
        return
    with st.expander("Governing formulas and parameter definitions", expanded=True):
        formula_column, definition_column = st.columns([0.90, 1.10])
        with formula_column:
            st.latex(r"h(T_m-T_{th})=\varepsilon\sigma(T_{th}^4-T_{sur}^4)")
            st.latex(r"T_m=T_{th}+\frac{\varepsilon\sigma}{h}(T_{th}^4-T_{sur}^4)")
            st.latex(r"Re_D=\frac{VD}{\nu},\qquad Nu_D=2+0.6Re_D^{1/2}Pr^{1/3},\qquad h=\frac{Nu_Dk_{air}}{D}")
            st.caption("Use kelvin in every fourth-power term. The correction assumes a known effective surrounding temperature and neglects conduction through the thermocouple leads.")
        with definition_column:
            st.dataframe(radiation_parameter_definitions(), use_container_width=True, hide_index=True)
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
        manual_h = st.number_input("Assumed natural-convection h for a 0.5 mm bead (W/m²·K)", min_value=1.0, max_value=250.0, step=1.0, key="rad_natural_h")
        st.caption("The controlled teaching data use a lower effective h for the 3 mm bead (half the 0.5 mm value) to represent its weaker area-based convective coupling.")
    h_scale = st.slider("h sensitivity multiplier", 0.50, 1.50, step=0.05, key="rad_h_scale", help="Tests sensitivity to property and correlation uncertainty.")
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
            h_base = float(manual_h) if diameter <= 0.5 else 0.5 * float(manual_h)
            reynolds, nusselt = np.nan, np.nan
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
    sample = correction.iloc[1]
    st.pyplot(
        radiation_h_sensitivity_plot(
            row,
            float(sample["Measured_bead_C"]),
            float(sample["Emissivity"]),
            float(sample["h_W_m2K"]),
        ),
        use_container_width=True,
    )
    st.caption("For a fixed bead and surrounding temperature, increasing h reduces the radiation correction. The h value where the blue curve meets the green T6 line is most consistent with the independent air reference.")
    with st.expander("Worked correction example: T8 black 0.5 mm", expanded=True):
        bead_kelvin = float(sample["Measured_bead_C"]) + 273.15
        surrounding_kelvin = float(row["T10_wall_C"]) + 273.15
        radiation_term = float(sample["Emissivity"]) * 5.670374419e-8 * (
            bead_kelvin**4 - surrounding_kelvin**4
        ) / float(sample["h_W_m2K"])
        st.latex(r"T_m=T_{th}+\frac{\varepsilon\sigma}{h}(T_{th}^4-T_{sur}^4)")
        st.write(
            f"Tth = {sample['Measured_bead_C']:.2f} °C = {bead_kelvin:.2f} K, Tsur = {row['T10_wall_C']:.2f} °C = "
            f"{surrounding_kelvin:.2f} K, ε = {sample['Emissivity']:.2f}, and h = {sample['h_W_m2K']:.2f} W/(m²·K)."
        )
        st.write(
            f"The radiation correction term is {radiation_term:.3f} K, giving Tm = {sample['Corrected_Tm_C']:.2f} °C. "
            f"The independent T6 reference is {sample['Reference_T6_C']:.2f} °C, so the residual is {sample['Residual_to_T6_K']:+.2f} K."
        )
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
    st.markdown(
        '<div class="equation-box"><b>Slide model:</b> R\'\'sample = L/k and ΔT = q\'\'R\'\'. Therefore ΔTtotal = q\'\'[R\'\'hot + L/k + R\'\'cold]. The colours below remain fixed: <span style="color:#C2410C"><b>hot contact</b></span>, <span style="color:#B28A00"><b>sample</b></span>, and <span style="color:#0F766E"><b>cold contact</b></span>.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.82, 1.18], gap="large")
    with left:
        q_flux = st.slider("Heat flux q'' (kW/m²)", 2.0, 80.0, step=1.0, key="cond_sandbox_q_flux", help="Increasing q'' increases every series temperature drop in direct proportion.")
        r_hot_micro = st.slider("Hot contact R'' (×10⁻⁴ m²·K/W)", 0.0, 20.0, step=0.2, key="cond_sandbox_r_hot", help="Changes only the red hot-interface jump when the other parameters are fixed.")
        conductivity = st.slider("Sample conductivity k (W/m·K)", 20, 240, step=5, key="cond_sandbox_k", help="Increasing k reduces the gold sample slope and temperature drop.")
        sample_length_mm = st.slider("Sample length L (mm)", 10, 60, step=1, key="cond_sandbox_length", help="Increasing L increases the gold sample resistance L/k.")
        r_cold_micro = st.slider("Cold contact R'' (×10⁻⁴ m²·K/W)", 0.0, 30.0, step=0.2, key="cond_sandbox_r_cold", help="Changes only the teal cold-interface jump when the other parameters are fixed.")
    q_flux_si = q_flux * 1000.0
    r_hot = r_hot_micro * 1e-4
    r_cold = r_cold_micro * 1e-4
    r_sample = (sample_length_mm / 1000.0) / conductivity
    drops = np.array([q_flux_si * r_hot, q_flux_si * r_sample, q_flux_si * r_cold])
    total = float(drops.sum())
    contact_share = 100.0 * (drops[0] + drops[2]) / total if total else 0.0
    with right:
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-bottom:10px">
              <div style="border:1px solid #FED7C3;border-left:5px solid {CONDUCTION_COLOURS['hot_contact']};border-radius:10px;padding:10px;background:#FFF8F4">
                <b style="color:{CONDUCTION_COLOURS['hot_contact']}">Hot contact</b><br>
                <span style="font-size:18px;font-weight:800">{drops[0]:.2f} K</span><br>
                <small>ΔTh = {q_flux_si:.0f} × {r_hot:.2e}</small>
              </div>
              <div style="border:1px solid #F3E6A8;border-left:5px solid {CONDUCTION_COLOURS['sample']};border-radius:10px;padding:10px;background:#FFFCED">
                <b style="color:{CONDUCTION_COLOURS['sample']}">Sample</b><br>
                <span style="font-size:18px;font-weight:800">{drops[1]:.2f} K</span><br>
                <small>ΔTs = {q_flux_si:.0f} × {sample_length_mm/1000:.3f}/{conductivity}</small>
              </div>
              <div style="border:1px solid #BFE3DD;border-left:5px solid {CONDUCTION_COLOURS['cold_contact']};border-radius:10px;padding:10px;background:#F2FBF9">
                <b style="color:{CONDUCTION_COLOURS['cold_contact']}">Cold contact</b><br>
                <span style="font-size:18px;font-weight:800">{drops[2]:.2f} K</span><br>
                <small>ΔTc = {q_flux_si:.0f} × {r_cold:.2e}</small>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cold_temperature = 25.0
        hot_temperature = cold_temperature + total
        after_hot_contact = hot_temperature - drops[0]
        before_cold_contact = after_hot_contact - drops[1]
        fig, (profile_ax, allocation_ax) = plt.subplots(2, 1, figsize=(9.2, 5.8), gridspec_kw={"height_ratios": [1.35, 1.0]})
        profile_ax.plot([0.04, 0.20], [hot_temperature, hot_temperature], color="#334155", lw=2.4)
        profile_ax.plot([0.20, 0.20], [hot_temperature, after_hot_contact], color=CONDUCTION_COLOURS["hot_contact"], lw=5, solid_capstyle="round")
        profile_ax.plot([0.20, 0.80], [after_hot_contact, before_cold_contact], color=CONDUCTION_COLOURS["sample"], lw=4)
        profile_ax.plot([0.80, 0.80], [before_cold_contact, cold_temperature], color=CONDUCTION_COLOURS["cold_contact"], lw=5, solid_capstyle="round")
        profile_ax.plot([0.80, 0.96], [cold_temperature, cold_temperature], color="#334155", lw=2.4)
        profile_ax.scatter([0.20, 0.20, 0.80, 0.80], [hot_temperature, after_hot_contact, before_cold_contact, cold_temperature], s=42, facecolor="white", edgecolor="#334155", zorder=5)
        profile_ax.set_xticks([0.12, 0.20, 0.50, 0.80, 0.88], ["Hot bar", "Hot\ncontact", "Sample", "Cold\ncontact", "Cold bar"])
        profile_ax.set_ylabel("Temperature (°C)")
        profile_ax.set_title("Live temperature path from the resistance-network slide", loc="left", fontweight="bold", color="#183A57")
        profile_ax.grid(axis="y", alpha=0.20)
        profile_ax.spines[["top", "right"]].set_visible(False)

        labels = ["Hot contact", "Sample", "Cold contact"]
        colours = [CONDUCTION_COLOURS["hot_contact"], CONDUCTION_COLOURS["sample"], CONDUCTION_COLOURS["cold_contact"]]
        bars = allocation_ax.barh(labels, drops, color=colours)
        allocation_ax.bar_label(bars, fmt="%.2f K", padding=3, fontsize=9)
        allocation_ax.set_xlabel("Temperature-drop contribution (K)")
        allocation_ax.set_title(f"Contact share = {contact_share:.1f}% of total ΔT = {total:.2f} K", loc="left", fontweight="bold", color="#183A57")
        allocation_ax.grid(axis="x", alpha=0.20)
        allocation_ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(h_pad=1.7)
        st.pyplot(fig, use_container_width=True)
        st.caption("Try one slider at a time: q'' changes all three drops; R''hot changes only the red jump; k or L changes the gold sample segment; R''cold changes only the teal jump.")
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
    st.markdown(
        '<div class="equation-box"><b>Slide model:</b> at steady state, convection and radiation balance: h(Tm - Tth) = εσ(Tth⁴ - Tsur⁴). The colours remain fixed: <span style="color:#0B4F8A"><b>air</b></span>, <span style="color:#C2410C"><b>bead</b></span>, <span style="color:#B42318"><b>wall</b></span>, <span style="color:#D97706"><b>radiation</b></span>, and <span style="color:#0F766E"><b>convection</b></span>.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.82, 1.18], gap="large")
    with left:
        air_c = st.slider("True air temperature Tm (°C)", 0.0, 80.0, step=1.0, key="rad_sandbox_air", help="The blue reference temperature that the experiment aims to measure.")
        wall_c = st.slider("Surrounding wall temperature Tsur (°C)", 0.0, 250.0, step=5.0, key="rad_sandbox_wall", help="A hotter red wall increases radiative heating of an exposed bead.")
        h = st.slider("Convection coefficient h (W/m²·K)", 2.0, 500.0, step=2.0, key="rad_sandbox_h", help="A larger h strengthens the teal convective link and pulls the bead toward the air temperature.")
        emissivity = st.slider("Bead emissivity ε", 0.05, 1.00, step=0.01, key="rad_sandbox_emissivity", help="A larger emissivity strengthens the orange radiative link to the surroundings.")
    bead_c = equilibrium_sensor_temperature_C(air_c, wall_c, h, emissivity)
    error = bead_c - air_c
    h_r = linearised_radiation_coefficient_W_m2K(bead_c, wall_c, emissivity)
    low_eps_bead = equilibrium_sensor_temperature_C(air_c, wall_c, h, 0.17)
    high_h_bead = equilibrium_sensor_temperature_C(air_c, wall_c, min(h * 4, 2000), emissivity)
    bead_kelvin = bead_c + 273.15
    air_kelvin = air_c + 273.15
    wall_kelvin = wall_c + 273.15
    radiation_to_bead = emissivity * 5.670374419e-8 * (wall_kelvin**4 - bead_kelvin**4)
    convection_from_bead = h * (bead_kelvin - air_kelvin)
    with right:
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-bottom:10px">
              <div style="border:1px solid #BDD7EA;border-left:5px solid {RADIATION_COLOURS['air']};border-radius:10px;padding:10px;background:#F2F8FC"><b style="color:{RADIATION_COLOURS['air']}">Air Tm</b><br><span style="font-size:18px;font-weight:800">{air_c:.2f} °C</span></div>
              <div style="border:1px solid #FED7C3;border-left:5px solid {RADIATION_COLOURS['bead']};border-radius:10px;padding:10px;background:#FFF8F4"><b style="color:{RADIATION_COLOURS['bead']}">Bead Tth</b><br><span style="font-size:18px;font-weight:800">{bead_c:.2f} °C</span><br><small>Bias = {error:+.2f} K</small></div>
              <div style="border:1px solid #F2C6C2;border-left:5px solid {RADIATION_COLOURS['wall']};border-radius:10px;padding:10px;background:#FFF6F5"><b style="color:{RADIATION_COLOURS['wall']}">Wall Tsur</b><br><span style="font-size:18px;font-weight:800">{wall_c:.2f} °C</span></div>
              <div style="border:1px solid #F5D7A4;border-left:5px solid {RADIATION_COLOURS['radiation']};border-radius:10px;padding:10px;background:#FFF9ED"><b style="color:{RADIATION_COLOURS['radiation']}">Radiation to bead</b><br><span style="font-size:17px;font-weight:800">{radiation_to_bead:+.2f} W/m²</span><br><small>εσ(Tsur⁴ - Tth⁴)</small></div>
              <div style="border:1px solid #BFE3DD;border-left:5px solid {RADIATION_COLOURS['convection']};border-radius:10px;padding:10px;background:#F2FBF9"><b style="color:{RADIATION_COLOURS['convection']}">Convection from bead</b><br><span style="font-size:17px;font-weight:800">{convection_from_bead:+.2f} W/m²</span><br><small>h(Tth - Tm)</small></div>
              <div style="border:1px solid #DCE4EC;border-left:5px solid {RADIATION_COLOURS['shield']};border-radius:10px;padding:10px;background:#F8FAFC"><b style="color:{RADIATION_COLOURS['shield']}">Coupling ratio</b><br><span style="font-size:17px;font-weight:800">hr/h = {h_r/h:.3f}</span><br><small>hr = {h_r:.2f} W/m²K</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8.7, 3.8))
        names = ["True air", "Current bead", "Polished ε=0.17", "4× stronger h", "Hot wall"]
        values = [air_c, bead_c, low_eps_bead, high_h_bead, wall_c]
        colours = [RADIATION_COLOURS["air"], RADIATION_COLOURS["bead"], RADIATION_COLOURS["shield"], RADIATION_COLOURS["convection"], RADIATION_COLOURS["wall"]]
        bars = ax.barh(names, values, color=colours)
        ax.bar_label(bars, fmt="%.1f °C", padding=3, fontsize=9)
        ax.set_xlabel("Temperature (°C)")
        ax.set_title("What changes the indicated temperature?", loc="left", fontweight="bold", color="#183A57")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("At equilibrium the signed orange radiative gain and teal convective loss match. Increase ε to strengthen radiation; increase h to strengthen convection and reduce the bead bias.")
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
        active_conduction_data(),
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
    analysed = analyse_radiation(active_radiation_data())
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
        analysed = analyse_conduction(active_conduction_data(), st.session_state.diameter_mm, st.session_state.heat_fraction)
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
        analysed = analyse_radiation(active_radiation_data())
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


def figure_png_bytes(figure: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def practical_report_bytes(
    practical: str,
    raw: pd.DataFrame,
    analysed: pd.DataFrame,
) -> bytes:
    student_details = {
        "name": st.session_state.student_name.strip() or "Not entered",
        "student_id": st.session_state.student_id.strip() or "Not entered",
        "group": st.session_state.group.strip() or "Not entered",
        "lab_date": str(st.session_state.lab_date),
        "pathway": st.session_state.pathway,
    }
    figures: list[tuple[str, bytes, str]] = []
    if practical == PRACTICAL_1:
        aim = (
            "Measure the steady one-dimensional temperature distribution, determine the sample thermal conductivity using Fourier's law, "
            "and quantify the temperature jumps and area-specific resistance at both contacts."
        )
        equations = [
            ("Q = fQ V I", "Electrical heat input retained by the one-dimensional model."),
            ("A = pi D^2 / 4 and q'' = Q/A", "Cross-sectional area and heat flux."),
            ("k = -q''/(dT/dx)", "Thermal conductivity from the fitted sample gradient."),
            ("Tface = m xinterface + b", "Interface temperature extrapolated from a regional fitted line."),
            ("R''c = (Tleft - Tright)/q''", "Area-specific thermal contact resistance."),
            (
                "(uk/k)^2 = (uV/V)^2 + (uI/I)^2 + (uL/L)^2 + (2uD/D)^2 + (uDeltaT/DeltaT)^2",
                "Independent input uncertainties combined by root-sum-of-squares; uDeltaT = sqrt(2) uT.",
            ),
        ]
        parameter_definitions = [
            (str(row.Symbol), f"{row.Definition}; unit: {row.Unit}")
            for row in conduction_parameter_definitions().itertuples(index=False)
        ]
        assumptions = [
            ("Diameter D", f"{st.session_state.diameter_mm:.2f} mm"),
            ("Heat fraction fQ", f"{st.session_state.heat_fraction:.3f}"),
            ("Brass reference k", f"{st.session_state.brass_reference_k:.1f} W/(m K); practical-note range 110-128 W/(m K)"),
            ("Aluminium reference k", f"{st.session_state.aluminium_reference_k:.1f} W/(m K); practical-note value approximately 180 W/(m K)"),
            (
                "Instrument uncertainties",
                f"uV={st.session_state.unc_v:.2f} V, uI={st.session_state.unc_i:.2f} A, uT={st.session_state.unc_t:.2f} K, "
                f"uD={st.session_state.unc_d:.2f} mm, uL={st.session_state.unc_l:.2f} mm",
            ),
        ]
        sample_calculation = []
        if not analysed.empty:
            result = analysed.iloc[0]
            source_row = raw.loc[int(result["Source_row"])]
            uncertainty_value = conduction_uncertainty_percent(
                result["Voltage_V"],
                result["Current_A"],
                float(source_row["T4_C"] - source_row["T5_C"]),
                st.session_state.diameter_mm,
                15.0,
                st.session_state.unc_v,
                st.session_state.unc_i,
                st.session_state.unc_t,
                st.session_state.unc_d,
                st.session_state.unc_l,
            )
            sample_calculation = [
                f"A = pi({st.session_state.diameter_mm/1000:.4f} m)^2/4 = {result['Area_m2']:.6e} m^2.",
                f"Q = {st.session_state.heat_fraction:.3f} x {result['Voltage_V']:.3f} V x {result['Current_A']:.3f} A = {result['Assumed_conduction_heat_W']:.3f} W; q'' = {result['Heat_flux_W_m2']:.1f} W/m^2.",
                f"ms = (T5-T4)/0.015 = ({source_row['T5_C']:.2f}-{source_row['T4_C']:.2f})/0.015 = {result['Sample_slope_K_m']:.2f} K/m.",
                f"k = -q''/ms = {result['Thermal_conductivity_W_mK']:.2f} W/(m K).",
                f"Hot contact temperatures are {result['Hot_bar_face_C']:.2f} C (left) and {result['Sample_hot_face_C']:.2f} C (right), giving R''hot = {result['Hot_contact_Rpp_m2K_W']:.3e} m^2 K/W.",
                f"Cold contact temperatures are {result['Sample_cold_face_C']:.2f} C (left) and {result['Cold_bar_face_C']:.2f} C (right), giving R''cold = {result['Cold_contact_Rpp_m2K_W']:.3e} m^2 K/W.",
                f"Using the stated independent input uncertainties, RSS propagation gives uk/k = {uncertainty_value:.2f}% (instrument terms only).",
            ]
            graph_result = analysed.loc[analysed["Contact_share_pct"].astype(float).idxmax()]
            graph_raw = raw.loc[int(graph_result["Source_row"])]
            figures = [
                (
                    f"Temperature-distance profile for {graph_result['Material']} trial {graph_result['Trial']}",
                    figure_png_bytes(conduction_temperature_plot(graph_raw, graph_result)),
                    "Separate regional fits expose the two interface temperatures on each contact and the unequal hot- and cold-contact jumps.",
                ),
                (
                    "Allocation of the fitted temperature drop",
                    figure_png_bytes(conduction_resistance_plot(graph_result)),
                    "The stacked contributions distinguish the specimen resistance from the two contact resistances.",
                ),
            ]
        discussion_notes = [
            ("Temperature distribution", st.session_state.cond_interpret_1),
            ("Importance of the contacts", st.session_state.cond_interpret_2),
            ("Trends across heat inputs and materials", st.session_state.cond_interpret_3),
            ("Model limitations and errors", st.session_state.cond_interpret_4),
        ]
        evidence = conduction_evidence()
        practical_code = "conduction"
    else:
        aim = (
            "Quantify radiation-induced thermocouple error under natural and forced convection, compare exposed and shielded cases, "
            "and apply a convection-radiation energy balance to estimate the medium temperature."
        )
        equations = [
            ("Error = Tbead - T6", "Difference between each bead reading and the upstream air reference."),
            ("h(Tm-Tth) = epsilon sigma (Tth^4-Tsur^4)", "Steady thermocouple energy balance; fourth-power temperatures are in kelvin."),
            ("Tm = Tth + (epsilon sigma/h)(Tth^4-Tsur^4)", "Radiation-corrected medium temperature."),
            ("ReD = V D/nu; NuD = 2 + 0.6 ReD^0.5 Pr^(1/3)", "Small-sphere forced-convection estimate."),
            ("h = NuD kair/D", "Convective heat-transfer coefficient."),
        ]
        parameter_definitions = [
            (str(row.Symbol), f"{row.Definition}; unit: {row.Unit}")
            for row in radiation_parameter_definitions().itertuples(index=False)
        ]
        assumptions = [
            ("T7", "0.5 mm polished bead, epsilon approximately 0.17"),
            ("T8", "0.5 mm black bead, epsilon approximately 0.98"),
            ("T9", "3 mm black bead, epsilon approximately 0.98"),
            ("Radiative surroundings", "For exposed cases, T10 is used as the effective surrounding temperature."),
            ("Lead conduction", "Neglected in the simple bead energy balance."),
        ]
        sample_calculation = []
        if not analysed.empty:
            forced_rows = analysed[pd.to_numeric(analysed["Air_velocity_m_s"], errors="coerce") > 0]
            result = forced_rows.iloc[0] if not forced_rows.empty else analysed.iloc[0]
            velocity = float(result["Air_velocity_m_s"])
            h_value = forced_convection_h(velocity, 0.5)[0] if velocity > 0 else float(st.session_state.rad_natural_h)
            corrected = radiation_corrected_medium_temperature_C(
                float(result["T8_small_black_C"]),
                float(result["T10_wall_C"]),
                h_value,
                0.98,
            )
            sample_calculation = [
                f"For {result['Case']}, T8 error = {result['T8_small_black_C']:.2f} - {result['T6_air_C']:.2f} = {result['T8_error_K']:.2f} K.",
                f"For the 0.5 mm bead at V = {velocity:.2f} m/s, h = {h_value:.2f} W/(m^2 K).",
                f"Using epsilon = 0.98 and Tsur = {result['T10_wall_C']:.2f} C, the corrected medium temperature is {corrected:.2f} C.",
                f"The correction residual relative to T6 is {corrected-float(result['T6_air_C']):+.2f} K.",
            ]
            exposed_forced = analysed[
                (pd.to_numeric(analysed["Air_velocity_m_s"], errors="coerce") > 0)
                & analysed["Shield"].astype(str).str.contains("exposed", case=False, na=False)
            ]
            graph_row = exposed_forced.iloc[0] if not exposed_forced.empty else result
            graph_velocity = float(graph_row["Air_velocity_m_s"])
            graph_h = forced_convection_h(graph_velocity, 0.5)[0] if graph_velocity > 0 else float(st.session_state.rad_natural_h)
            figures = [
                (
                    "Thermocouple bias across the four operating cases",
                    figure_png_bytes(radiation_error_plot(analysed)),
                    "The controlled data isolate the expected effects: black and larger beads are more radiation-sensitive, while shielding and stronger airflow reduce bias.",
                ),
                (
                    f"Sensitivity of the T8 correction to h for {graph_row['Case']}",
                    figure_png_bytes(
                        radiation_h_sensitivity_plot(
                            graph_row,
                            float(graph_row["T8_small_black_C"]),
                            0.98,
                            graph_h,
                        )
                    ),
                    "The inferred medium temperature depends strongly on h when convection is weak; at larger h the correction approaches the measured bead temperature.",
                ),
            ]
        discussion_notes = [
            ("Error ranking", st.session_state.rad_interpret_1),
            ("Shield and airflow effects", st.session_state.rad_interpret_2),
            ("Emissivity and bead size", st.session_state.rad_interpret_3),
            ("Correction and model limitations", st.session_state.rad_interpret_4),
        ]
        evidence = radiation_evidence()
        practical_code = "radiation"

    return build_practical_report(
        practical_code=practical_code,
        practical_title=PRACTICAL_TITLES[practical],
        student_details=student_details,
        raw_data=raw,
        analysed_data=analysed,
        aim=aim,
        equations=equations,
        parameter_definitions=parameter_definitions,
        assumptions=assumptions,
        sample_calculation=sample_calculation,
        evidence=evidence,
        discussion_notes=discussion_notes,
        figures=figures,
        logo_path=ASSET_DIR / "jcu_logo.jpg",
    )


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
        raw = active_conduction_data().copy()
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
        raw = active_radiation_data().copy()
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
    st.subheader("Generate the practical report")
    st.caption("The Word report includes JCU branding, student details, equations and definitions, raw and analysed data, important graphs, one worked example, evidence, and your discussion notes.")
    report_key = f"_generated_report_{prefix}"
    report_filename_key = f"_generated_report_filename_{prefix}"
    if st.button("Generate practical report (.docx)", type="primary", use_container_width=True, key=f"generate_report_{prefix}"):
        try:
            st.session_state[report_key] = practical_report_bytes(practical, raw, analysed)
            st.session_state[report_filename_key] = f"ME3512_{prac_part}_{student_part}_report.docx"
            st.success("The Word practical report has been generated from the current session values.")
        except Exception as error:
            st.error(f"The report could not be generated: {error}")
    if st.session_state.get(report_key):
        st.download_button(
            "Download generated Word report",
            data=st.session_state[report_key],
            file_name=st.session_state.get(report_filename_key, f"ME3512_{prac_part}_{student_part}_report.docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key=f"download_report_{prefix}",
        )
        st.caption("Regenerate the report after changing measurements, assumptions, or discussion notes so the Word file uses the latest values.")
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
