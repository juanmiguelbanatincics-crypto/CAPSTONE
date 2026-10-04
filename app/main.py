from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_clone import clone_footer, clone_section, clone_topbar, install_clone_style
from ui_clone_table import clone_table

# Ensure the sibling app/data.py wins over the project data/namespace directory
# regardless of whether Streamlit is launched from the project root or app/.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data import (
    get_database_path,
    get_annual_inventory_profile,
    get_core_table_health,
    get_demand_behavior_summary,
    get_executive_snapshot,
    get_inventory_demand_explorer,
    get_latest_abc_behavior_matrix,
    get_latest_abc_profile,
    get_medicine_annual_history,
    get_system_context,

    FORECAST_EVIDENCE_TABLES,
    get_forecast_profile,
    get_forecast_summary,
    get_forecast_by_abc,
    get_forecast_by_behavior,
    get_forecast_evidence_table,

    PROCUREMENT_EVIDENCE_TABLES,
    get_procurement_evidence_table,
    get_procurement_medicine_inputs,
    get_procurement_readiness_summary,
    get_procurement_table_schema,
    get_procurement_optimizer_options,
    get_procurement_optimizer_result,

    RETROSPECTIVE_EVIDENCE_TABLES,
    get_retrospective_core_tables,
    get_retrospective_evidence_table,
    get_retrospective_table_schema,

    EXPIRY_EVIDENCE_TABLES,
    get_expiry_dss_summary,
    get_expiry_evidence_table,
    get_expiry_fact_records,
    get_expiry_medicine_explorer,
    get_expiry_records_for_medicine,
    get_expiry_table_schema,

    GOVERNANCE_EVIDENCE_TABLES,
    get_governance_evidence_table,
    get_governance_overview,
    get_governance_table_health,
    get_governance_table_schema,
)


# ============================================================
# READ-ONLY UI CACHE
# ============================================================

# The dashboard reads immutable, local DuckDB evidence. Cache reader results
# in the Streamlit process so switching a page, tab, filter, or scenario does
# not repeatedly reopen the database and rebuild the same DataFrames.
def cache_reader(function):
    return st.cache_data(
        ttl=600,
        show_spinner=False,
    )(function)


get_system_context = cache_reader(get_system_context)
get_core_table_health = cache_reader(get_core_table_health)
get_executive_snapshot = cache_reader(get_executive_snapshot)
get_annual_inventory_profile = cache_reader(
    get_annual_inventory_profile
)
get_demand_behavior_summary = cache_reader(
    get_demand_behavior_summary
)
get_latest_abc_behavior_matrix = cache_reader(
    get_latest_abc_behavior_matrix
)
get_latest_abc_profile = cache_reader(get_latest_abc_profile)
get_inventory_demand_explorer = cache_reader(
    get_inventory_demand_explorer
)
get_medicine_annual_history = cache_reader(
    get_medicine_annual_history
)
get_forecast_profile = cache_reader(get_forecast_profile)
get_forecast_summary = cache_reader(get_forecast_summary)
get_forecast_by_abc = cache_reader(get_forecast_by_abc)
get_forecast_by_behavior = cache_reader(get_forecast_by_behavior)
get_forecast_evidence_table = cache_reader(
    get_forecast_evidence_table
)
get_procurement_evidence_table = cache_reader(
    get_procurement_evidence_table
)
get_procurement_medicine_inputs = cache_reader(
    get_procurement_medicine_inputs
)
get_procurement_readiness_summary = cache_reader(
    get_procurement_readiness_summary
)
get_procurement_table_schema = cache_reader(
    get_procurement_table_schema
)
get_procurement_optimizer_options = cache_reader(
    get_procurement_optimizer_options
)
get_procurement_optimizer_result = cache_reader(
    get_procurement_optimizer_result
)
get_retrospective_core_tables = cache_reader(
    get_retrospective_core_tables
)
get_retrospective_evidence_table = cache_reader(
    get_retrospective_evidence_table
)
get_retrospective_table_schema = cache_reader(
    get_retrospective_table_schema
)
get_expiry_dss_summary = cache_reader(get_expiry_dss_summary)
get_expiry_evidence_table = cache_reader(get_expiry_evidence_table)
get_expiry_fact_records = cache_reader(get_expiry_fact_records)
get_expiry_medicine_explorer = cache_reader(
    get_expiry_medicine_explorer
)
get_expiry_records_for_medicine = cache_reader(
    get_expiry_records_for_medicine
)
get_expiry_table_schema = cache_reader(get_expiry_table_schema)
get_governance_evidence_table = cache_reader(
    get_governance_evidence_table
)
get_governance_overview = cache_reader(get_governance_overview)
get_governance_table_health = cache_reader(
    get_governance_table_health
)
get_governance_table_schema = cache_reader(
    get_governance_table_schema
)


# ============================================================
# CURRENT SYSTEM CONTEXT
# ============================================================

SYSTEM_CONTEXT = get_system_context()

HISTORICAL_START_YEAR = SYSTEM_CONTEXT[
    "Historical_Start_Year"
]

LATEST_COMPLETED_YEAR = SYSTEM_CONTEXT[
    "Latest_Completed_Year"
]

PLANNING_YEAR = SYSTEM_CONTEXT[
    "Planning_Year"
]


HISTORICAL_YEAR_COUNT = (
    LATEST_COMPLETED_YEAR
    -
    HISTORICAL_START_YEAR
    +
    1
)


HISTORICAL_PERIOD_LABEL = (
    f"{HISTORICAL_START_YEAR}–{LATEST_COMPLETED_YEAR}"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CDH Pharmaceutical Inventory DSS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown(
    """
<style>

/* Shared visual system: presentation-only; page layout and data behavior stay intact. */
:root {
    --dss-ink: #132b4a;
    --dss-text: #344a66;
    --dss-muted: #687b93;
    --dss-line: #dce5ed;
    --dss-surface: #ffffff;
    --dss-canvas: #f7fafc;
    --dss-teal: #1f7a72;
    --dss-teal-dark: #145e59;
    --dss-teal-soft: #e6f4f1;
    --dss-blue-soft: #edf5ff;
    --dss-success: #237a59;
    --dss-warning: #a66a1f;
}

html,
body,
[class*="css"] {
    color: var(--dss-text);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 100% 0%, #eef7f6 0, transparent 25rem),
        var(--dss-canvas);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

[data-testid="stHorizontalBlock"] > div {
    min-width: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fbfc 0%, #f1f6f7 100%);
    border-right: 1px solid var(--dss-line);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.65rem 0.9rem 1.25rem;
}

[data-testid="stSidebar"] .stCaption {
    color: var(--dss-muted);
    font-size: 0.75rem;
    line-height: 1.45;
}

[data-testid="stSidebar"] hr {
    border-color: var(--dss-line);
    margin: 1.15rem 0;
}


.dss-eyebrow {
    color: var(--dss-teal);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}


.dss-title {
    color: var(--dss-ink);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.12;
    margin-top: 0.25rem;
    margin-bottom: 0.4rem;
}


.dss-subtitle {
    color: var(--dss-muted);
    font-size: 0.98rem;
    line-height: 1.5;
    margin-top: 0;
    margin-bottom: 1rem;
}


.guardrail {
    background: linear-gradient(90deg, #eff8f6 0%, #f7fbfb 100%);
    border: 1px solid #cce8e1;
    border-left: 4px solid var(--dss-teal);
    border-radius: 10px;
    color: var(--dss-text);
    box-shadow: 0 3px 12px rgba(20, 94, 89, 0.04);
    line-height: 1.55;
    margin-top: 1.25rem;
    padding: 0.9rem 1.1rem;
}

.guardrail-title { display: block; }
.guardrail-body { margin-top: 0.4rem; }
.dss-svg-icon { display: inline-block; flex: 0 0 auto; height: 20px; max-height: 20px; max-width: 20px; width: 20px; }
.dss-sidebar-icon { height: 16px; max-height: 16px; max-width: 16px; width: 16px; }

.dss-sidebar-brand { color: var(--dss-ink); font-size: 1.13rem; font-weight: 800; letter-spacing: -0.025em; }
.dss-nav-group { color: var(--dss-teal); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.12em; margin: 0.4rem 0 0.45rem; text-transform: uppercase; }
.dss-nav-group.evidence { color: var(--dss-muted); }
.dss-sidebar-panel { background: rgba(255,255,255,0.82); border: 1px solid var(--dss-line); border-radius: 10px; color: var(--dss-text); font-size: 0.8rem; line-height: 1.5; margin: 0.8rem 0; padding: 0.8rem 0.85rem; }
.dss-sidebar-status { color: var(--dss-success); font-size: 0.8rem; font-weight: 700; }
.dss-card { background: var(--dss-surface); border: 1px solid var(--dss-line); border-radius: 11px; box-shadow: 0 6px 18px rgba(19,43,74,0.045); display: flex; flex-direction: column; height: 100%; min-height: 246px; padding: 17px 18px; transition: box-shadow .18s ease, transform .18s ease; }
.dss-card:hover { box-shadow: 0 10px 24px rgba(19,43,74,0.08); transform: translateY(-2px); }
.dss-card-label { color: var(--dss-muted); font-size: 0.76rem; font-weight: 750; letter-spacing: 0.02em; text-transform: uppercase; }
.dss-card-value { color: var(--dss-ink); font-size: 1.82rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.15; margin: 9px 0 10px; min-height: 2.1rem; }
.dss-status { align-items: center; display: inline-flex; font-size: 0.73rem; font-weight: 750; gap: 0.35rem; min-height: 1.4rem; }
.dss-card-description { color: var(--dss-muted); font-size: 0.76rem; line-height: 1.45; margin-top: auto; min-height: 3.35rem; padding-top: 12px; }
.dss-status-message { align-items: center; background: #f1f7f7; border: 1px solid #d3e8e5; border-left: 3px solid #52727a; border-radius: 8px; color: var(--dss-text); display: flex; font-size: 0.84rem; gap: 0.5rem; line-height: 1.45; margin: 0.35rem 0; padding: 0.72rem 0.8rem; }
.dss-question-icon { align-items: center; display: inline-flex; margin-right: 0.35rem; vertical-align: -0.2rem; }
.dss-question-icon svg { height: 18px; width: 18px; }

/* Navigation and form controls share one spacing, border, and focus language. */
[data-testid="stSidebar"] [data-testid="stRadio"] label { align-items: center; border-left: 3px solid transparent; border-radius: 8px; color: var(--dss-text); font-size: 0.84rem; font-weight: 600; margin: 0.12rem 0; min-height: 38px; padding: 0.5rem 0.65rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #eaf4f2; color: var(--dss-teal-dark); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: var(--dss-teal-soft); border-left-color: var(--dss-teal); color: var(--dss-teal-dark); font-weight: 750; }
[data-testid="stSidebar"] [data-testid="stRadio"] input { display: none; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.45rem; border-bottom: 1px solid var(--dss-line); }
[data-testid="stTabs"] button { color: var(--dss-muted); font-size: 0.82rem; font-weight: 700; height: 2.55rem; padding: 0 0.9rem; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--dss-teal); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--dss-teal); height: 3px; }

[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea {
    background: #ffffff;
    border-color: #cfdbe6;
    border-radius: 8px;
    box-shadow: none;
    color: var(--dss-ink);
    min-height: 2.45rem;
}

[data-baseweb="select"] > div:hover,
[data-testid="stTextInput"] input:hover,
[data-testid="stNumberInput"] input:hover,
[data-testid="stDateInput"] input:hover,
[data-testid="stTextArea"] textarea:hover {
    border-color: #8ebdb6;
}

[data-baseweb="select"] > div:focus-within,
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--dss-teal);
    box-shadow: 0 0 0 3px rgba(31,122,114,0.13);
}

[data-testid="stWidgetLabel"] p { color: var(--dss-text); font-size: 0.78rem; font-weight: 700; }
[data-testid="stButton"] > button { background: var(--dss-teal); border: 1px solid var(--dss-teal); border-radius: 8px; color: #ffffff; font-weight: 700; min-height: 2.35rem; padding: 0.35rem 0.9rem; }
[data-testid="stButton"] > button:hover { background: var(--dss-teal-dark); border-color: var(--dss-teal-dark); color: #ffffff; }
[data-testid="stButton"] > button:focus { box-shadow: 0 0 0 3px rgba(31,122,114,0.16); }

[data-testid="stExpander"] { background: #ffffff; border: 1px solid var(--dss-line); border-radius: 9px; box-shadow: 0 2px 8px rgba(19,43,74,0.025); }
[data-testid="stExpander"] summary { color: var(--dss-ink); font-size: 0.84rem; font-weight: 750; }
[data-testid="stDataFrame"], [data-testid="stTable"] { background: #ffffff; border: 1px solid var(--dss-line); border-radius: 9px; overflow: hidden; }
[data-testid="stDataFrame"] * { color: var(--dss-ink); }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--dss-line); border-radius: 10px; height: 100%; min-height: 108px; padding: 0.9rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--dss-muted); font-size: 0.75rem; font-weight: 750; }
[data-testid="stMetricValue"] { color: var(--dss-ink); font-weight: 800; letter-spacing: -0.025em; }
[data-testid="stAlert"] { border-radius: 9px; }
[data-testid="stProgress"] [role="progressbar"] > div { background-color: var(--dss-teal) !important; }
[data-testid="stProgress"] > div > div > div { background-color: var(--dss-teal) !important; }
[data-testid="stDivider"] { border-color: var(--dss-line); }
[data-testid="stMarkdownContainer"] h3 { color: var(--dss-ink); font-size: 1.1rem; font-weight: 800; letter-spacing: -0.015em; }

/* Reference dashboard skin: UI presentation only. */
:root {
    --dss-ink: #06275d;
    --dss-text: #153969;
    --dss-muted: #607795;
    --dss-line: #dce4eb;
    --dss-surface: #ffffff;
    --dss-canvas: #f6f8fa;
    --dss-teal: #167784;
    --dss-teal-dark: #0f6673;
    --dss-teal-soft: #dcecef;
    --dss-success: #198460;
    --dss-warning: #bf7913;
}

[data-testid="stAppViewContainer"] { background: #f6f8fa; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.block-container {
    max-width: none;
    padding: 1.45rem 3.7rem 3.5rem;
}

[data-testid="stSidebar"] {
    background: #f0f4f6;
    border-right: 1px solid #d7e0e6;
    height: 100vh !important;
    min-width: 21.875rem !important;
    top: 0 !important;
    width: 21.875rem !important;
}

[data-testid="stSidebar"] > div:first-child {
    box-sizing: border-box;
    height: 100%;
    min-width: 21.875rem;
    padding: 1.7rem 0.95rem 1.4rem;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    display: flex;
    flex-direction: column;
    min-height: 100%;
}

[data-testid="stSidebar"] hr {
    border-color: #d9e2e7;
    margin: 1.35rem -0.95rem 1.25rem;
}

.dss-brand-row { align-items: flex-start; display: flex; gap: 0.75rem; margin-left: 0.6rem; }
.dss-brand-mark { align-items: center; background: #1d7782; border-radius: 0.6rem; color: #ffffff; display: inline-flex; font-size: 1rem; font-weight: 800; height: 2.5rem; justify-content: center; letter-spacing: -0.05em; width: 2.5rem; }
.dss-sidebar-brand { color: #06275d; font-size: 1.25rem; font-weight: 800; line-height: 1.35; margin-top: 0.54rem; }
.dss-sidebar-product { color: #55708f; font-size: 0.875rem; line-height: 1.35; margin-top: 0.58rem; }
.dss-nav-group { color: #637895; font-size: 0.75rem; letter-spacing: 0.12em; margin: 0.25rem 0 0.6rem 0.9rem; }
.dss-nav-group.evidence { color: #637895; margin-top: 2rem; }

/* One real button per navigation row prevents duplicate radios/icons. */
[class*="st-key-sidebar-nav-"] { margin: 0.15rem 0; }
[class*="st-key-sidebar-nav-"] button {
    align-items: center;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-left: 3px solid transparent !important;
    border-radius: 0.6rem !important;
    box-shadow: none !important;
    box-sizing: border-box;
    color: #163b6a !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    height: 3.45rem !important;
    justify-content: flex-start !important;
    padding: 0 0.75rem !important;
    text-align: left;
    width: 100%;
}

[class*="st-key-sidebar-nav-"] button:hover {
    background: #e6eff1 !important;
    color: #0f7080 !important;
}

[class*="st-key-sidebar-nav-"] button[kind="primary"] {
    background: #d9e8ea !important;
    border-left-color: #117c89 !important;
    color: #0f7080 !important;
    font-weight: 700 !important;
}

[class*="st-key-sidebar-nav-"] button::before {
    color: #5f7b9a;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1;
    margin-right: 0.78rem;
}
[class*="st-key-sidebar-nav-planning-overview"] button::before { content: "◉"; }
[class*="st-key-sidebar-nav-medicine-prioritization"] button::before { content: "⬡"; }
[class*="st-key-sidebar-nav-forecast-planning"] button::before { content: "△"; }
[class*="st-key-sidebar-nav-procurement-planning"] button::before { content: "⊞"; }
[class*="st-key-sidebar-nav-expiry-and-data-quality-review"] button::before { content: "⊛"; }
[class*="st-key-sidebar-nav-retrospective-validation"] button::before { content: "◐"; }
[class*="st-key-sidebar-nav-methodology-and-governance"] button::before { content: "□"; }
[class*="st-key-sidebar-nav-"] button[kind="primary"]::before { color: #0f7d8b; }

.dss-sidebar-panel { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.85rem; color: #173863; font-size: 0.875rem; line-height: 1.58; margin: 0.8rem 0; padding: 0.95rem 1rem; }
[class*="st-key-sidebar-footer"] { margin-top: auto; padding-top: 1rem; }
.dss-sidebar-panel.context { margin-top: 0; }
.dss-context-heading { color: #0b315e; font-weight: 800; margin-bottom: 0.45rem; }
.dss-context-row { display: flex; justify-content: space-between; margin-top: 0.22rem; }
.dss-context-row span { color: #5d7592; }
.dss-context-row strong { color: #092e5a; font-weight: 700; }
.dss-system-status { align-items: center; color: #0d725b; display: flex; font-size: 0.75rem; gap: 0.45rem; margin-top: 0.16rem; }
.dss-system-dot { background: #54aa8b; border-radius: 50%; height: 0.58rem; width: 0.58rem; }

/* Compact sidebar rhythm: keep the reference footer anchored, reduce blank menu space. */
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 0.8rem 0.8rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.2rem; }
[data-testid="stSidebar"] hr { margin: 1rem -0.8rem 0.8rem; }
.dss-brand-row { gap: 0.65rem; margin-left: 0.45rem; }
.dss-brand-mark { height: 2.35rem; width: 2.35rem; }
.dss-sidebar-brand { margin-top: 0.38rem; }
.dss-sidebar-product { margin-top: 0.35rem; }
.dss-nav-group { margin: 0.15rem 0 0.35rem 0.8rem; }
.dss-nav-group.evidence { margin-top: 1.05rem; }
[class*="st-key-sidebar-nav-"] { margin: 0.04rem 0; }
[class*="st-key-sidebar-nav-"] button { font-size: 0.96rem !important; height: 2.95rem !important; padding: 0 0.72rem !important; }
[class*="st-key-sidebar-nav-"] button::before { margin-right: 0.7rem; }
[class*="st-key-sidebar-footer"] { padding-top: 0.7rem; }
.dss-sidebar-panel { line-height: 1.45; margin: 0.5rem 0; padding: 0.75rem 0.9rem; }

.dss-page-header { margin-bottom: 2.3rem; }
.dss-title { color: #06275d; font-size: 2.28rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.12; margin: 0; }
.dss-subtitle { color: #5d7592; font-size: 1rem; line-height: 1.5; margin: 0.45rem 0 0; }

.dss-card { border-color: #dce4eb; border-radius: 0.85rem; box-shadow: 0 1px 2px rgba(8, 35, 73, 0.035); min-height: 9rem; padding: 1.15rem 1.35rem; }
.dss-card:hover { box-shadow: 0 4px 12px rgba(8, 35, 73, 0.07); transform: translateY(-1px); }
.dss-card-label { color: #58708f; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.02em; }
.dss-card-value { color: #06275d; font-size: 2.48rem; line-height: 1.08; margin: 0.45rem 0 0; min-height: auto; }
.dss-card .dss-status { display: none; }
.dss-card-description { color: #597291; font-size: 0.82rem; line-height: 1.35; margin-top: auto; min-height: 0; padding-top: 0.25rem; }

[class*="st-key-question-panel-"] {
    background: #ffffff;
    border: 1px solid #dce4eb;
    border-radius: 0.95rem;
    box-shadow: 0 1px 2px rgba(8, 35, 73, 0.025);
    margin-bottom: 1.85rem;
    padding: 0.1rem 1.55rem 1.45rem;
}

[class*="st-key-question-panel-"] [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-question-row-"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-color: #dce4eb;
    border-radius: 0.95rem;
    box-shadow: 0 1px 3px rgba(8, 35, 73, 0.04);
}

.dss-question-panel-title { color: #082e5b; font-size: 1.04rem; font-weight: 800; padding: 1rem 0 0.95rem; }
.dss-question-row-icon { align-items: center; background: #edf5f6; border-radius: 0.65rem; display: inline-flex; height: 2.9rem; justify-content: center; width: 2.9rem; }
.dss-question-row-icon svg { height: 1.35rem; width: 1.35rem; }
[class*="st-key-question-action-"] button { background: transparent !important; border: 0 !important; box-shadow: none !important; color: #06275d !important; font-size: 1rem !important; justify-content: flex-start; min-height: 2.25rem !important; padding: 0 !important; text-align: left; }
[class*="st-key-question-action-"] button:hover { background: transparent !important; color: #0d7080 !important; }
.dss-question-description { color: #5c7390; font-size: 0.86rem; line-height: 1.45; margin-top: -0.35rem; }
.dss-badge { align-items: center; background: #fff5ce; border: 1px solid #efc84f; border-radius: 999px; color: #b76e09; display: inline-flex; font-size: 0.79rem; font-weight: 700; gap: 0.35rem; justify-content: center; margin-top: 0.25rem; min-height: 2.15rem; padding: 0 0.8rem; white-space: nowrap; }
.dss-badge.ok { background: #d9fae7; border-color: #6ce4af; color: #14724f; }
.dss-badge.unavailable { background: #eef3f7; border-color: #d2dde7; color: #637895; }

.dss-readiness-card { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 1px 2px rgba(8,35,73,0.025); margin: 0 0 1.85rem; padding: 1.55rem; }
.dss-readiness-heading { color: #082e5b; font-size: 1rem; font-weight: 800; }
.dss-readiness-summary { align-items: baseline; color: #5c7390; display: flex; font-size: 0.94rem; justify-content: space-between; margin-top: 1rem; }
.dss-readiness-summary strong { color: #082e5b; font-size: 0.94rem; }
.dss-readiness-track { background: #dfe5e9; border-radius: 999px; height: 0.82rem; margin-top: 0.45rem; overflow: hidden; }
.dss-readiness-fill { background: #197986; border-radius: inherit; height: 100%; }
.dss-readiness-note { color: #5c7390; font-size: 0.86rem; margin-top: 0.7rem; }

.dss-executive-note { background: #fff9e7; border-left: 0.32rem solid #c47c19; border-radius: 0.9rem; color: #23426d; margin: 0 0 1.85rem; padding: 1.18rem 1.45rem; }
.dss-executive-note-title { color: #bd7514; font-size: 0.95rem; font-weight: 800; }
.dss-executive-note-body { font-size: 0.93rem; line-height: 1.55; margin-top: 0.3rem; }

.dss-decision-status { color: #082e5b; font-size: 1.08rem; font-weight: 800; margin: 0.55rem 0 0.95rem; }
.dss-status-card { border: 1px solid #efc84f; border-radius: 0.9rem; box-sizing: border-box; min-height: 10.25rem; padding: 1.5rem; }
.dss-status-card.attention { background: #fff4ca; color: #b86f0a; }
.dss-status-card.ok { background: #d9fae8; border-color: #62dfa9; color: #147452; }
.dss-status-card-value { font-size: 2.35rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1; }
.dss-status-card-value:empty { display: none; }
.dss-status-card-title { font-size: 0.94rem; font-weight: 800; margin-top: 0.45rem; }
.dss-status-card-body { font-size: 0.86rem; line-height: 1.45; margin-top: 0.35rem; opacity: 0.78; }

.dss-breakdown-card { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 1px 2px rgba(8,35,73,0.025); margin-top: 1.85rem; overflow: hidden; }
.dss-breakdown-heading { border-bottom: 1px solid #dce4eb; color: #082e5b; font-size: 1.08rem; font-weight: 800; padding: 1.2rem 1.55rem; }
.dss-breakdown-table { border-collapse: collapse; font-size: 0.9rem; width: 100%; }
.dss-breakdown-table th { background: #f4f6f8; color: #627895; font-size: 0.76rem; font-weight: 800; letter-spacing: 0.025em; padding: 1rem 1.25rem; text-align: left; text-transform: uppercase; }
.dss-breakdown-table th.number, .dss-breakdown-table td.number { text-align: right; }
.dss-breakdown-table td { border-top: 1px solid #edf1f4; color: #082e5b; padding: 1rem 1.25rem; }
.dss-breakdown-table tr:nth-child(even) td { background: #fbfcfd; }
.dss-table-pill { border-radius: 999px; display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 0.35rem 0.7rem; white-space: nowrap; }
.dss-table-pill.good { background: #d9fae8; border: 1px solid #62dfa9; color: #147452; }
.dss-table-pill.attention { background: #fff4ca; border: 1px solid #efc84f; color: #b86f0a; }
.dss-table-pill.neutral { background: #f3f6f8; border: 1px solid #d5e0e8; color: #617791; }

/* Compact Executive Overview: reduce only inter-section space. */
.dss-page-header { margin-bottom: 1.15rem; }
[class*="st-key-question-panel-"] { margin-bottom: 1rem; padding: 0.1rem 1.35rem 1.05rem; }
.dss-question-panel-title { padding: 0.85rem 0 0.7rem; }
.dss-question-row-icon { height: 2.9rem; width: 2.9rem; }
.dss-readiness-card { margin-bottom: 1rem; padding: 1.15rem 1.35rem; }
.dss-readiness-summary { margin-top: 0.7rem; }
.dss-readiness-track { height: 0.82rem; margin-top: 0.45rem; }
.dss-readiness-note { margin-top: 0.5rem; }
.dss-executive-note { margin-bottom: 1rem; padding: 0.95rem 1.25rem; }
.dss-executive-note-body { font-size: 0.93rem; line-height: 1.55; margin-top: 0.3rem; }
.dss-card { box-sizing: border-box; height: 11.25rem; min-height: 11.25rem; padding: 1.15rem 1.35rem; }
.dss-card-label { min-height: 1.8rem; }
.dss-card-value { min-height: 2.7rem; white-space: nowrap; }
.dss-decision-status { margin: 0.15rem 0 0.65rem; }
.dss-breakdown-card { margin-top: 1rem; }

/* Methodology & Governance reference layout. */
.dss-governance-intro { background: #edf5ff; border-left: 0.3rem solid #2876c5; border-radius: 0.9rem; color: #294d7e; margin: 0 0 1rem; padding: 1.05rem 1.35rem; }
.dss-governance-intro-title { color: #2265b1; font-size: 0.96rem; font-weight: 800; }
.dss-governance-intro-body { font-size: 0.91rem; line-height: 1.52; margin: 0.28rem 0 0 1.35rem; }
.dss-governance-panel { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 1px 2px rgba(8,35,73,0.025); margin: 0 0 1rem; overflow: hidden; }
.dss-governance-panel-title { border-bottom: 1px solid #dce4eb; color: #082e5b; font-size: 1.08rem; font-weight: 800; padding: 1.15rem 1.35rem; }
.dss-method-body { color: #294d7e; font-size: 0.91rem; line-height: 1.55; padding: 0.8rem 1.35rem 1rem 3.9rem; }
.dss-method-row { align-items: center; border-top: 1px solid #e8eef2; color: #082e5b; display: flex; font-size: 0.96rem; font-weight: 700; min-height: 4.7rem; padding: 0 1.35rem; }
.dss-method-row.first { border-top: 0; min-height: 4.55rem; }
.dss-method-number { align-items: center; background: #1f7d88; border-radius: 0.48rem; color: #ffffff; display: inline-flex; font-size: 0.86rem; font-weight: 800; height: 1.9rem; justify-content: center; margin-right: 1rem; min-width: 1.9rem; }
.dss-method-action { color: #627895; font-size: 1.1rem; font-weight: 500; margin-left: auto; }
.dss-governance-table-wrap { overflow-x: auto; padding: 1.25rem 1.35rem; }
.dss-governance-table { border-collapse: collapse; font-size: 0.91rem; width: 100%; }
.dss-governance-table th { background: #f4f6f8; border-bottom: 1px solid #dce4eb; color: #627895; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.025em; padding: 1rem 1.25rem; text-align: left; text-transform: uppercase; }
.dss-governance-table td { border-bottom: 1px solid #e9eef2; color: #294d7e; padding: 0.95rem 1.25rem; }
.dss-governance-table tr:nth-child(even) td { background: #fbfcfd; }
.dss-governance-table td:first-child { color: #082e5b; font-weight: 700; width: 24%; }
.dss-timeline { padding: 0.2rem 1.35rem 1.1rem; }
.dss-timeline-item { display: grid; grid-template-columns: 2.8rem 1fr; min-height: 7.25rem; position: relative; }
.dss-timeline-item:last-child { min-height: 5.2rem; }
.dss-timeline-marker { align-items: center; display: flex; flex-direction: column; position: relative; }
.dss-timeline-marker::after { background: #dbe5ea; bottom: 0; content: ""; left: 50%; position: absolute; top: 2.85rem; transform: translateX(-50%); width: 2px; }
.dss-timeline-item:last-child .dss-timeline-marker::after { display: none; }
.dss-timeline-number { align-items: center; background: #1f7d88; border-radius: 50%; color: #ffffff; display: flex; font-size: 0.87rem; font-weight: 800; height: 2.45rem; justify-content: center; width: 2.45rem; z-index: 1; }
.dss-timeline-content { padding: 0.2rem 0 0 0.8rem; }
.dss-timeline-month { color: #0f7180; font-size: 0.8rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
.dss-timeline-title { color: #082e5b; font-size: 0.98rem; font-weight: 700; margin-top: 0.25rem; }
.dss-timeline-owner { color: #627895; font-size: 0.88rem; margin-top: 0.28rem; }
.dss-evidence-card { border-left: 0.3rem solid #2876c5; border-radius: 0.85rem; box-sizing: border-box; min-height: 9rem; padding: 1rem 1.2rem; }
.dss-evidence-card.source { background: #edf5ff; color: #294d7e; }
.dss-evidence-card.validation { background: #effbf4; border-left-color: #25835d; color: #216047; }
.dss-evidence-card-title { font-size: 0.95rem; font-weight: 800; }
.dss-evidence-card-body { font-size: 0.9rem; line-height: 1.5; margin-top: 0.35rem; }

/* Procurement reference layout. */
.dss-card-value.teal { color: #157985; }
.dss-card-value.danger { color: #ca3d32; }
.dss-procurement-banner { background: #eff9fa; border: 1px solid #a9dbe0; border-radius: 0.9rem; color: #082e5b; font-size: 0.95rem; line-height: 1.55; margin: 1rem 0; padding: 1.1rem 1.3rem; }
.dss-procurement-banner strong { font-weight: 800; }
.dss-procurement-table-card { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 1px 2px rgba(8,35,73,0.025); margin: 1rem 0; overflow: hidden; }
.dss-procurement-table-title { border-bottom: 1px solid #dce4eb; color: #082e5b; font-size: 1.08rem; font-weight: 800; padding: 1.1rem 1.35rem; }
.dss-procurement-table-wrap { overflow-x: auto; padding: 1.1rem 1.35rem 1.35rem; }
.dss-procurement-table { border-collapse: collapse; font-size: 0.88rem; min-width: 59rem; width: 100%; }
.dss-procurement-table th { background: #f4f6f8; border-bottom: 1px solid #dce4eb; color: #627895; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.02em; padding: 0.92rem 0.85rem; text-align: left; text-transform: uppercase; white-space: nowrap; }
.dss-procurement-table th.number, .dss-procurement-table td.number { text-align: right; }
.dss-procurement-table td { border-bottom: 1px solid #e9eef2; color: #082e5b; padding: 0.9rem 0.85rem; white-space: nowrap; }
.dss-procurement-table tr:nth-child(even) td { background: #fbfcfd; }
.dss-procurement-table .funded { color: #087f63; font-weight: 800; }
.dss-procurement-table .unfunded { color: #d6342d; font-weight: 800; }
.dss-procurement-table .budget-level { color: #147783; font-weight: 800; }
[class*="st-key-procurement-allocation-preview"] { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 1px 2px rgba(8,35,73,0.025); margin: 1rem 0; padding: 0.15rem 1.35rem 1.2rem; }
.dss-procurement-allocation-title { color: #082e5b; font-size: 1.08rem; font-weight: 800; margin: 0.9rem 0 0.85rem; }
.dss-procurement-search-count { color: #5c7390; font-size: 0.86rem; font-weight: 600; padding-top: 1.9rem; }
.dss-procurement-detail-heading { color: #082e5b; font-size: 1.03rem; font-weight: 800; margin: 1rem 0 0.55rem; }

/* Planning Overview and Medicine Prioritization only.  These selectors are
   intentionally page-scoped so Forecasting, Procurement, Expiry, Governance,
   their charts, and their interactions keep their existing presentation. */
.dss-workspace-page-header { align-items: flex-end; display: flex; flex-wrap: wrap; gap: 1.25rem; justify-content: space-between; margin: 0.1rem 0 1.45rem; }
.dss-workspace-eyebrow { color: #147783; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
.dss-workspace-heading { color: #102a59; font-size: clamp(2rem, 3vw, 2.6rem); font-weight: 800; letter-spacing: -0.045em; line-height: 1.1; margin: 0.55rem 0 0; }
.dss-workspace-subtitle { color: #66778a; font-size: 1.02rem; line-height: 1.48; margin: 0.48rem 0 0; max-width: 53rem; }
.dss-planning-year-chip { align-items: center; background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.72rem; color: #66778a; display: inline-flex; font-size: 0.9rem; gap: 0.55rem; min-height: 2.95rem; padding: 0 0.9rem; white-space: nowrap; }
.dss-planning-year-chip strong { color: #102a59; font-weight: 800; }
.dss-planning-year-chip svg { height: 1.1rem; width: 1.1rem; }

[class*="st-key-question-panel-planning"],
[class*="st-key-question-panel-medicine_review"] { background: #ffffff; border: 1px solid #dce4eb; border-radius: 1rem; box-shadow: 0 5px 16px rgba(16,42,89,0.035); margin-bottom: 1rem; padding: 0.1rem 1.55rem 1.2rem; }
[class*="st-key-question-panel-planning"] .dss-question-panel-title,
[class*="st-key-question-panel-medicine_review"] .dss-question-panel-title { color: #102a59; font-size: 1.08rem; padding: 1.05rem 0 0.82rem; }
[class*="st-key-question-row-planning-"],
[class*="st-key-question-row-medicine_review-"] { margin: 0.15rem 0 0; }
[class*="st-key-question-row-planning-"] [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-question-row-medicine_review-"] [data-testid="stVerticalBlockBorderWrapper"] { background: #f7fafb; border-color: #dce5ea; box-shadow: none; }
[class*="st-key-question-panel-planning"] .dss-question-description,
[class*="st-key-question-panel-medicine_review"] .dss-question-description { color: #69798b; font-size: 0.95rem; }

[class*="st-key-planning-metric-grid"],
[class*="st-key-priority-metric-grid"] { margin: 0.95rem 0 1rem; }
.dss-workspace-metric { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 4px 14px rgba(16,42,89,0.03); box-sizing: border-box; display: flex; flex-direction: column; min-height: 12.15rem; padding: 1.25rem 1.35rem; }
.dss-workspace-metric-top { align-items: center; display: flex; gap: 0.7rem; justify-content: space-between; min-height: 2.35rem; }
.dss-workspace-metric-icon { align-items: center; background: #eaf4f5; border-radius: 0.62rem; display: inline-flex; height: 2.45rem; justify-content: center; width: 2.45rem; }
.dss-workspace-metric-icon svg { height: 1.25rem; width: 1.25rem; }
.dss-workspace-badge { border-radius: 999px; display: inline-flex; font-size: 0.76rem; font-weight: 700; line-height: 1; padding: 0.45rem 0.66rem; white-space: nowrap; }
.dss-workspace-badge.info { background: #edf4fb; border: 1px solid #d8e5f1; color: #2870bc; }
.dss-workspace-badge.attention { background: #fff8ec; border: 1px solid #efd8ad; color: #9b6314; }
.dss-workspace-badge.success { background: #ebf8f0; border: 1px solid #cfead9; color: #27825d; }
.dss-workspace-metric-label { color: #65778a; font-size: 0.95rem; font-weight: 650; margin-top: 0.95rem; }
.dss-workspace-metric-value { color: #102a59; font-size: clamp(2rem, 2.7vw, 2.55rem); font-weight: 800; letter-spacing: -0.045em; line-height: 1.05; margin-top: 0.5rem; overflow-wrap: anywhere; }
.dss-workspace-metric-value.teal { color: #147783; }
.dss-workspace-metric-description { color: #6a7a8b; font-size: 0.9rem; line-height: 1.4; margin-top: auto; padding-top: 0.65rem; }

.dss-planning-readiness { align-items: center; background: #f7fafb; border: 1px solid #dce5ea; border-radius: 0.8rem; display: grid; gap: 1.15rem; grid-template-columns: minmax(15rem, 0.85fr) minmax(19rem, 1.15fr); margin: 1rem 0; padding: 1.45rem 1.6rem; }
.dss-planning-readiness-summary { align-items: center; display: grid; gap: 0.7rem; grid-template-columns: 2.65rem 1fr auto; }
.dss-planning-readiness-icon { align-items: center; background: #eaf4f5; border-radius: 50%; display: inline-flex; height: 2.65rem; justify-content: center; width: 2.65rem; }
.dss-planning-readiness-icon svg { height: 1.25rem; width: 1.25rem; }
.dss-planning-readiness-label { color: #102a59; font-size: 0.98rem; font-weight: 800; }
.dss-planning-readiness-caption { color: #6a7a8b; font-size: 0.84rem; margin-top: 0.17rem; }
.dss-planning-readiness-score { color: #147783; font-size: clamp(2rem, 3vw, 2.7rem); font-weight: 800; letter-spacing: -0.05em; line-height: 1; }
.dss-planning-readiness-track-area { border-left: 1px solid #dfe7eb; padding-left: 1.35rem; }
.dss-planning-readiness-track-head { align-items: baseline; color: #102a59; display: flex; font-size: 0.93rem; font-weight: 650; justify-content: space-between; gap: 1rem; }
.dss-planning-readiness-track-head strong { color: #147783; font-size: 1rem; }
.dss-planning-readiness-track { background: #e0e7eb; border-radius: 999px; height: 0.62rem; margin-top: 0.8rem; overflow: hidden; }
.dss-planning-readiness-fill { background: #1d7b84; border-radius: inherit; height: 100%; }
.dss-planning-readiness-legend { color: #68798a; display: flex; flex-wrap: wrap; font-size: 0.82rem; gap: 1.1rem; margin-top: 0.58rem; }
.dss-planning-readiness-legend span::before { background: #1d7b84; border-radius: 50%; content: ""; display: inline-block; height: 0.52rem; margin-right: 0.38rem; width: 0.52rem; }
.dss-planning-readiness-legend span:last-child::before { background: #dda63b; }

[class*="st-key-planning-economic-profile"],
[class*="st-key-planning-evidence-coverage"],
[class*="st-key-priority-classification-profile"],
[class*="st-key-priority-historical-profile"],
[class*="st-key-priority-matrix-profile"],
[class*="st-key-priority-explorer"] { background: #ffffff; border: 1px solid #dce4eb; border-radius: 0.95rem; box-shadow: 0 3px 12px rgba(16,42,89,0.025); margin: 1rem 0; padding: 0.3rem 1.2rem 1.2rem; }
.dss-workspace-section-title { color: #102a59; font-size: 1.1rem; font-weight: 800; margin: 0.85rem 0 0.2rem; }
.dss-workspace-section-caption { color: #6a7a8b; font-size: 0.92rem; line-height: 1.45; margin: 0 0 0.9rem; }
.dss-workspace-chart-note { border-top: 1px solid #e4eaee; color: #67788a; font-size: 0.86rem; line-height: 1.42; margin-top: 0.25rem; padding-top: 0.85rem; }

.st-key-planning-decision-status { margin: 1rem 0; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] { background: #ffffff; border: 0; border-radius: 1rem; box-shadow: 0 3px 12px rgba(16,42,89,0.025); }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details { border: 1px solid #dce4eb; border-radius: 1rem; overflow: hidden; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary { background: #ffffff; border-bottom: 1px solid #dce4eb; min-height: 5.6rem; padding: 1.75rem 1.9rem; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary [data-testid="stMarkdownContainer"] p { color: #172d54; font-family: Arial, sans-serif; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.035em; line-height: 1.2; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary > span { display: flex; width: 100%; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary > span > span:first-child { color: #687a8c; order: 2; margin-left: auto; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > [data-testid="stExpanderDetails"] { padding: 1.9rem; }
.planning-actions-intro { align-items: center; color: #6b7b8c; display: flex; font-family: "Segoe UI", Arial, sans-serif; font-size: 1.08rem; gap: 0.75rem; justify-content: space-between; line-height: 1.5; margin-bottom: 0.35rem; }
.st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > [data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"] { gap: 1rem; }
.planning-actions-intro span:last-child { color: #172d54; font-size: 0.94rem; white-space: nowrap; }
.st-key-planning-action-grid > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { gap: 1.25rem; align-items: stretch; }
.st-key-planning-action-grid [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
.st-key-planning-action-grid [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] { height: 100%; }
[class*="st-key-planning-action-card-"] { background: #ffffff; border: 1px solid #dce4eb !important; border-radius: 0.65rem; box-shadow: none; box-sizing: border-box; gap: 0.75rem; height: 100%; min-height: 25.9rem; padding: 1.55rem !important; }
.planning-action-top { align-items: center; display: flex; justify-content: space-between; gap: 0.5rem; }
.planning-action-top > svg { height: 1.45rem; max-height: none; max-width: none; width: 1.45rem; }
.planning-action-badge { align-items: center; background: #fff9ef; border: 1px solid #efd8b2; border-radius: 999px; color: #986015; display: inline-flex; font-family: "Segoe UI", Arial, sans-serif; font-size: 0.96rem; gap: 0.4rem; line-height: 1.2; padding: 0.46rem 0.7rem; white-space: nowrap; }
.planning-action-badge.success { background: #edf7f1; border-color: #cfe7d9; color: #28855e; }
.planning-action-badge svg { height: 0.95rem; width: 0.95rem; }
.planning-action-title { color: #172d54; font-family: "Segoe UI", Arial, sans-serif; font-size: 1.4rem; font-weight: 650; letter-spacing: -0.025em; line-height: 1.3; margin: 1.05rem 0 0; }
.planning-action-description { color: #6b7b8c; font-family: "Segoe UI", Arial, sans-serif; font-size: 1.07rem; line-height: 1.6; margin: 1.1rem 0 0.45rem; }
.planning-action-meta { align-items: center; background: #f6f8fa; border-radius: 0.65rem; color: #6b7b8c; display: flex; font-family: "Segoe UI", Arial, sans-serif; font-size: 1.03rem; gap: 0.7rem; line-height: 1.45; margin-top: 0.75rem; min-height: 3.5rem; padding: 0.8rem 1rem; }
.planning-action-meta svg { height: 1.05rem; width: 1.05rem; }
.st-key-planning-action-review-rows { gap: 0.6rem; margin-top: 0.25rem; }
[class*="st-key-planning-action-review-row-"] button { align-items: center; background: #fff9ef !important; border: 1px solid #efd8b2 !important; border-radius: 0.6rem !important; box-shadow: none !important; color: #172d54 !important; display: flex; font-family: "Segoe UI", Arial, sans-serif; font-weight: 400 !important; justify-content: space-between !important; min-height: 3.3rem !important; padding: 0.65rem 0.9rem !important; text-align: left; width: 100%; }
[class*="st-key-planning-action-review-row-"] button::after { color: #117989; content: "Review now  →"; font-size: 0.88rem; font-weight: 550; margin-left: 0.65rem; white-space: nowrap; }
[class*="st-key-planning-action-review-row-"] button p { font-family: "Segoe UI", Arial, sans-serif; font-size: 0.96rem; }
[class*="st-key-planning-action-review-row-"] button strong { font-size: 1.2rem; margin-right: 0.35rem; }
[class*="st-key-planning-action-footer-"] { margin-top: auto; padding-top: 0.4rem; }
[class*="st-key-planning-action-card-"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-planning-action-footer-"]) { margin-top: auto; }
[class*="st-key-planning-action-footer-"] button { align-items: center; background: #ffffff !important; border: 1px solid #dce4eb !important; border-radius: 0.6rem !important; box-shadow: none !important; color: #117989 !important; display: flex; font-family: "Segoe UI", Arial, sans-serif; font-weight: 500 !important; justify-content: space-between !important; min-height: 3.1rem !important; padding: 0.6rem 1.2rem !important; text-align: left; width: 100%; }
[class*="st-key-planning-action-footer-"] button p { font-family: "Segoe UI", Arial, sans-serif; font-size: 1.08rem; font-weight: 500; }
[class*="st-key-planning-action-footer-"] button::after { content: "→"; font-size: 1.45rem; line-height: 1; margin-left: 0.7rem; }
[class*="st-key-planning-action-footer-"] button > div,
[class*="st-key-planning-action-review-row-"] button > div { display: flex; justify-content: flex-start; flex: 1 1 auto; min-width: 0; }
[class*="st-key-planning-action-footer-"] button [data-testid="stMarkdownContainer"],
[class*="st-key-planning-action-review-row-"] button [data-testid="stMarkdownContainer"] { text-align: left; }
.planning-actions-intro, .planning-action-badge, .planning-action-title,
.planning-action-description, .planning-action-meta,
[class*="st-key-planning-action-footer-"] button p,
[class*="st-key-planning-action-review-row-"] button p { font-family: Arial, sans-serif; }
[class*="st-key-planning-action-footer-"] button:hover,
[class*="st-key-planning-action-review-row-"] button:hover { background: #edf6f7 !important; border-color: #87b8bf !important; }
[class*="st-key-planning-action-footer-"] button:focus-visible,
[class*="st-key-planning-action-review-row-"] button:focus-visible { outline: 2px solid #147783; outline-offset: 2px; }
.st-key-planning-action-details { border-top: 1px solid #e2e9ee; margin-top: 0.75rem; padding-top: 1rem; }
@media (max-width: 1350px) {
    .st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > [data-testid="stExpanderDetails"] { padding: 1.2rem; }
    [class*="st-key-planning-action-card-"] { padding: 1.1rem !important; }
    .planning-action-title { font-size: 1.2rem; }
    .planning-action-description { font-size: 0.96rem; }
    .planning-action-badge { font-size: 0.83rem; }
    [class*="st-key-planning-action-review-row-"] button::after { content: "→"; }
}
@media (max-width: 1350px) {
    .st-key-planning-action-grid > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-planning-action-grid > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100% !important; min-width: 0 !important; width: 100% !important; }
    [class*="st-key-planning-action-card-"] { min-height: 0; }
    [class*="st-key-planning-action-footer-"] { padding-top: 1rem; }
    [class*="st-key-planning-action-review-row-"] button::after { content: "Review now  →"; }
}
@media (max-width: 600px) {
    .st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary { min-height: 4.3rem; padding: 1.2rem; }
    .st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > summary [data-testid="stMarkdownContainer"] p { font-size: 1.4rem; }
    .st-key-planning-decision-status > [data-testid="stLayoutWrapper"] > [data-testid="stExpander"] > details > [data-testid="stExpanderDetails"] { padding: 0.9rem; }
    .planning-actions-intro { align-items: flex-start; flex-direction: column; font-size: 0.94rem; }
    .planning-action-meta { font-size: 0.9rem; }
    [class*="st-key-planning-action-review-row-"] button::after { content: "→"; }
}
[class*="st-key-planning-readiness-breakdown"] .dss-breakdown-card { margin-top: 1rem; }

[class*="st-key-priority-classification-profile"] .dss-workspace-section-title,
[class*="st-key-priority-historical-profile"] .dss-workspace-section-title,
[class*="st-key-priority-matrix-profile"] .dss-workspace-section-title,
[class*="st-key-priority-explorer"] .dss-workspace-section-title { font-size: 1.16rem; }
.dss-priority-chart-title { color: #102a59; font-size: 1rem; font-weight: 750; margin: 0.4rem 0 0.25rem; }
.dss-priority-explorer-count { color: #617386; font-size: 0.9rem; font-weight: 650; padding-top: 0.7rem; text-align: right; }

/* Keep the existing routes, labels, and seven buttons; only remove the unused
   bottom anchoring that caused the oversized blank sidebar gap. */
[class*="st-key-sidebar-footer"] { margin-top: 1rem; padding-top: 0.35rem; }

@media (max-width: 900px) {
    .dss-workspace-page-header { align-items: flex-start; gap: 0.85rem; margin-bottom: 1rem; }
    .dss-planning-year-chip { width: 100%; }
    .dss-workspace-subtitle { font-size: 0.95rem; }
    .dss-workspace-metric { min-height: 10.6rem; padding: 1.05rem 1.1rem; }
    .dss-workspace-metric-value { white-space: normal; }
    .dss-planning-readiness { grid-template-columns: 1fr; padding: 1.1rem; }
    .dss-planning-readiness-track-area { border-left: 0; border-top: 1px solid #dfe7eb; padding-left: 0; padding-top: 1rem; }
    [class*="st-key-question-panel-planning"],
    [class*="st-key-question-panel-medicine_review"],
    [class*="st-key-planning-economic-profile"],
    [class*="st-key-planning-evidence-coverage"],
    [class*="st-key-priority-classification-profile"],
    [class*="st-key-priority-historical-profile"],
    [class*="st-key-priority-matrix-profile"],
    [class*="st-key-priority-explorer"] { padding-left: 0.9rem; padding-right: 0.9rem; }
    .dss-priority-explorer-count { padding-top: 0.15rem; text-align: left; }
}

@media (max-width: 900px) {
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { min-width: 0 !important; width: auto !important; }
    .block-container { padding: 1.35rem 1rem 2rem; }
    .dss-title { font-size: 1.85rem; }
    .dss-card-value { font-size: 1.55rem; }
    .dss-card { height: auto; min-height: 9rem; }
}

</style>
""",
unsafe_allow_html=True,
)


install_clone_style()

# ============================================================
# NAVIGATION
# ============================================================

OPERATIONAL_PAGES = [
    "Planning Overview",
    "Medicine Prioritization",
    "Forecast Planning",
    "Procurement Planning",
    "Expiry & Data Quality Review",
]

EVIDENCE_PAGES = [
    "Retrospective Validation",
    "Methodology & Governance",
]

PAGES = OPERATIONAL_PAGES + EVIDENCE_PAGES

PAGE_ICON_NAMES = {
    "Planning Overview": "overview",
    "Medicine Prioritization": "medicine",
    "Forecast Planning": "forecast",
    "Procurement Planning": "procurement",
    "Expiry & Data Quality Review": "expiry",
    "Retrospective Validation": "validation",
    "Methodology & Governance": "governance",
}

SVG_ICONS = {
    "overview": '<circle cx="10" cy="10" r="7"/><path d="M10 6v4l3 2"/>',
    "medicine": '<rect x="3" y="4" width="14" height="12" rx="2"/><path d="M6 8h8M6 12h5"/>',
    "forecast": '<path d="M3 15l4-5 3 2 5-7"/><path d="M12 5h3v3"/>',
    "procurement": '<path d="M4 5h12l-1 9H5L4 5Z"/><path d="M7 5a3 3 0 0 1 6 0M7 17h.01M13 17h.01"/>',
    "expiry": '<rect x="3" y="4" width="14" height="13" rx="2"/><path d="M6 2v4M14 2v4M3 8h14M7 11h3M7 14h5"/>',
    "validation": '<path d="M10 2l2 3 3 .5-2 2.5.5 3.5-3.5-1.5L6.5 11 7 8.5 5 6l3-.5L10 2Z"/><path d="m8 15 2 2 4-4"/>',
    "governance": '<path d="M10 2 17 5v5c0 4-3 6.5-7 8-4-1.5-7-4-7-8V5l7-3Z"/><path d="m7 10 2 2 4-4"/>',
    "check": '<path d="m4 10 4 4 8-8"/>',
    "attention": '<path d="M10 3 18 17H2L10 3Z"/><path d="M10 8v4M10 14h.01"/>',
    "info": '<circle cx="10" cy="10" r="7"/><path d="M10 9v5M10 6h.01"/>',
}


def icon_svg(
    name: str,
    color: str = "#1f6f78",
    size: int = 20,
    class_name: str = "dss-svg-icon",
) -> str:
    return (
        f'<svg class="{class_name}" width="{size}" height="{size}" '
        f'viewBox="0 0 20 20" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">'
        f'{SVG_ICONS.get(name, SVG_ICONS["info"])}'
        '</svg>'
    )


CHART_COLORS = [
    "#1f6f78",
    "#52727a",
    "#a66a1f",
    "#8a9ba8",
]


def style_chart(fig):
    fig.update_layout(
        colorway=CHART_COLORS,
        font={"family": "Inter, Segoe UI Variable, Segoe UI, sans-serif", "color": "#344054"},
        height=320,
        margin={"l": 12, "r": 12, "t": 44, "b": 12},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9e2e7")
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f3", zeroline=False)
    return fig


def _activate_operational_page() -> None:
    st.session_state["active_page"] = st.session_state["operational_page"]


def _activate_evidence_page() -> None:
    st.session_state["active_page"] = st.session_state["evidence_page"]


def _activate_page(page_name: str) -> None:
    """Navigate through the sidebar without introducing a second radio state."""
    st.session_state["active_page"] = page_name


def sidebar_navigation_button(page_name: str) -> None:
    """Render one full-width, aligned sidebar navigation row."""
    key_slug = (
        page_name.lower()
        .replace("&", "and")
        .replace(" ", "-")
    )
    st.button(
        page_name,
        key=f"sidebar-nav-{key_slug}",
        type=(
            "primary"
            if st.session_state["active_page"] == page_name
            else "secondary"
        ),
        use_container_width=True,
        on_click=_activate_page,
        args=(page_name,),
    )


if "active_page" not in st.session_state:
    st.session_state["active_page"] = OPERATIONAL_PAGES[0]


if st.session_state["active_page"] in OPERATIONAL_PAGES:
    st.session_state["operational_page"] = st.session_state["active_page"]
    st.session_state["evidence_page"] = None
else:
    st.session_state["evidence_page"] = st.session_state["active_page"]
    st.session_state["operational_page"] = None


with st.sidebar:

    st.markdown(
        '<div class="dss-brand-row">'
        '<div class="dss-brand-mark">Rx</div>'
        '<div><div class="dss-sidebar-brand">CDH Inventory DSS</div>'
        '<div class="dss-sidebar-product">Pharmaceutical decision<br>'
        'support</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="clone-sidebar-workspace">Hospital pharmacy workspace</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="dss-nav-group">Operational</div>',
        unsafe_allow_html=True,
    )

    for navigation_page in OPERATIONAL_PAGES:
        sidebar_navigation_button(navigation_page)

    st.markdown(
        '<div class="dss-nav-group evidence">Analytical Evidence &amp; Governance</div>',
        unsafe_allow_html=True,
    )

    for navigation_page in EVIDENCE_PAGES:
        sidebar_navigation_button(navigation_page)


    with st.container(key="sidebar-footer"):
        st.markdown(
            '<div class="dss-sidebar-panel context">'
            '<div class="dss-context-heading">Planning Context</div>'
            '<div class="dss-context-row"><span>Planning year</span>'
            f'<strong>{PLANNING_YEAR}</strong></div>'
            '<div class="dss-context-row"><span>Historical evidence</span>'
            f'<strong>{HISTORICAL_PERIOD_LABEL}</strong></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="dss-sidebar-panel"><strong>System Status</strong>'
            '<div class="dss-system-status"><span class="dss-system-dot"></span>'
            'Database connected</div></div>',
            unsafe_allow_html=True,
        )

    page = st.session_state["active_page"]

clone_topbar(page)

# ============================================================
# COMMON COMPONENTS
# ============================================================


def page_header(
    eyebrow: str | None,
    title: str,
    subtitle: str,
) -> None:

    header_html = f"""
<div class="dss-page-header">
<div class="dss-title">{title}</div>
<div class="dss-subtitle">{subtitle}</div>
</div>
"""

    if eyebrow:
        header_html = (
            f'<div class="dss-eyebrow">{eyebrow}</div>'
            + header_html
        )

    st.markdown(
        header_html,
        unsafe_allow_html=True,
    )


def workspace_page_header(
    title: str,
    subtitle: str,
) -> None:
    """Page-local header used only by the two planning workspace views."""
    st.markdown(
        f"""
<section class="dss-workspace-page-header">
  <div>
    <div class="dss-workspace-eyebrow">{PLANNING_YEAR} Planning Workspace</div>
    <div class="dss-workspace-heading">{title}</div>
    <div class="dss-workspace-subtitle">{subtitle}</div>
  </div>
  <div class="dss-planning-year-chip">
    {icon_svg("expiry", "#61758b", 18)}
    <span>Planning year</span><strong>{PLANNING_YEAR}</strong>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def workspace_metric_card(
    label: str,
    value: str,
    description: str,
    icon: str,
    badge: str,
    badge_tone: str = "info",
    value_tone: str = "default",
    footer_label: str | None = None,
    footer_href: str | None = None,
) -> None:
    """Presentation-only metric card with live values supplied by its caller."""
    footer = (
        f'<a class="clone-metric-link" href="{html.escape(footer_href, quote=True)}" '
        f'target="_self">{html.escape(footer_label)}<span aria-hidden="true">›</span></a>'
        if footer_label and footer_href else ""
    )
    st.markdown(
        f"""
<section class="dss-workspace-metric">
  <div class="dss-workspace-metric-top">
    <span class="dss-workspace-metric-icon">{icon_svg(icon, "#147783", 20)}</span>
    <span class="dss-workspace-badge {badge_tone}">{badge}</span>
  </div>
  <div class="dss-workspace-metric-label">{label}</div>
  <div class="dss-workspace-metric-value {value_tone}">{value}</div>
  <div class="dss-workspace-metric-description">{description}</div>
  {footer}
</section>
""",
        unsafe_allow_html=True,
    )


def workspace_section_header(
    title: str,
    caption: str,
) -> None:
    st.markdown(
        f'<div class="dss-workspace-section-title">{title}</div>'
        f'<div class="dss-workspace-section-caption">{caption}</div>',
        unsafe_allow_html=True,
    )


def planning_donut_chart_card(
    key: str,
    title: str,
    caption: str,
    labels: list[str],
    values: list[int],
    colors: dict[str, str],
    note: str,
) -> None:
    """Render a small, live planning profile chart without changing data rules."""
    chart_data = pd.DataFrame({"Label": labels, "Medicines": values})
    total = int(sum(values))
    with st.container(border=True, key=key):
        workspace_section_header(title, caption)
        figure = px.pie(
            chart_data,
            names="Label",
            values="Medicines",
            color="Label",
            color_discrete_map=colors,
        )
        figure.update_traces(
            hole=0.74,
            textinfo="none",
            marker={"line": {"color": "#ffffff", "width": 4}},
            hovertemplate=(
                "%{label}: %{value:,} medicines (%{percent})<extra></extra>"
            ),
        )
        figure.update_layout(
            height=228,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font={"family": "Inter, sans-serif", "color": "#14264a"},
            annotations=[
                {
                    "text": f"<b>{total:,}</b><br><span style='font-size:14px'>medicines</span>",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 27, "color": "#14264a"},
                }
            ],
        )
        chart_col, legend_col = st.columns([1.05, 1.65], vertical_alignment="center")
        with chart_col:
            st.plotly_chart(
                figure, width="stretch", config={"displayModeBar": False}
            )
        with legend_col:
            legend_rows = []
            for label, value in zip(labels, values):
                share = value / total * 100 if total else 0.0
                legend_rows.append(
                    '<div class="clone-donut-legend-row">'
                    f'<i style="background:{colors[label]}"></i>'
                    f'<span>{html.escape(label)}</span>'
                    f'<strong>{int(value):,}</strong><span>{share:.1f}%</span>'
                    '</div>'
                )
            st.markdown(
                '<div class="clone-donut-legend">' + ''.join(legend_rows) + '</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="dss-workspace-chart-note">'
            f'{icon_svg("info", "#147783", 17)}{note}</div>',
            unsafe_allow_html=True,
        )


def workspace_readiness_card(
    readiness_score: float,
    optimizer_ready: int,
    planning_medicines: int,
) -> None:
    """Recompose existing readiness values without changing their calculation."""
    readiness_pct = max(0.0, min(readiness_score * 100, 100.0))
    optimizer_ready_pct = (
        optimizer_ready / planning_medicines * 100
        if planning_medicines
        else 0
    )
    remaining = max(planning_medicines - optimizer_ready, 0)
    st.markdown(
        f"""
<section class="dss-planning-readiness">
  <div class="dss-planning-readiness-summary">
    <span class="dss-planning-readiness-icon">{icon_svg("overview", "#147783", 20)}</span>
    <div>
      <div class="dss-planning-readiness-label">Planning readiness</div>
      <div class="dss-planning-readiness-caption">Preparation score, not medicine coverage</div>
    </div>
    <div class="dss-planning-readiness-score">{readiness_pct:.0f}%</div>
  </div>
  <div class="dss-planning-readiness-track-area">
    <div class="dss-planning-readiness-track-head">
      <span>{optimizer_ready:,} of {planning_medicines:,} medicines are optimizer-ready</span>
      <strong>{optimizer_ready_pct:.1f}%</strong>
    </div>
    <div class="dss-planning-readiness-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{optimizer_ready_pct:.0f}">
      <div class="dss-planning-readiness-fill" style="width:{optimizer_ready_pct:.2f}%;"></div>
    </div>
    <div class="dss-planning-readiness-legend">
      <span>Ready: {optimizer_ready:,}</span>
      <span>Needs review: {remaining:,}</span>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

def section_title(
    title: str,
    subtitle: str | None = None,
) -> None:

    st.subheader(
        title
    )

    if subtitle:

        st.caption(
            subtitle
        )



def dashboard_card(
    label: str,
    value: str,
    status: str = "Based on annual forecast",
    description: str | None = None,
    value_tone: str = "default",
) -> None:


    status_colors = {

        "Planning Ready":
            "#2f7d5c",

        "Review Needed":
            "#a66a1f",

        "Based on annual forecast":
            "#1f6f78",

        "Historical evidence available":
            "#52727a",

    }


    color = (
        status_colors
        .get(
            status,
            "#1f6f78"
        )
    )


    st.markdown(
        f"""
<div class="dss-card">
<div class="dss-card-label">{label}</div>
<div class="dss-card-value {value_tone}">{value}</div>
<div class="dss-status" style="color:{color};">
{icon_svg('check' if status == 'Planning Ready' else 'attention' if status == 'Review Needed' else 'info', color)}
{status}
</div>
<div class="dss-card-description">{description or ""}</div>
</div>
""",
        unsafe_allow_html=True,
    )



def decision_guardrail(
    title: str,
    message: str,
) -> None:


    st.markdown(
        f"""
<div class="guardrail">

<strong class="guardrail-title">
{title}
</strong>

<div class="guardrail-body">
{message}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


def status_message(message: str, severity: str = "info") -> None:
    colors = {
        "ok": "#2f7d5c",
        "attention": "#a66a1f",
        "info": "#52727a",
    }
    color = colors.get(severity, colors["info"])
    icon = "check" if severity == "ok" else "attention" if severity == "attention" else "info"
    st.markdown(
        f'<div class="dss-status-message" style="border-left-color:{color};">'
        f'{icon_svg(icon, color)}{message}</div>',
        unsafe_allow_html=True,
    )


def readiness_progress_card(
    readiness_score: float,
    optimizer_ready: int,
    planning_medicines: int,
) -> None:
    """Present the existing readiness calculation in the dashboard card UI."""
    percent = max(0.0, min(readiness_score * 100, 100.0))
    remaining = max(planning_medicines - optimizer_ready, 0)
    st.markdown(
        f"""
<section class="dss-readiness-card">
  <div class="dss-readiness-heading">Overall Planning Readiness</div>
  <div class="dss-readiness-summary">
    <span>{percent:.0f}% ready — {optimizer_ready:,} of {planning_medicines:,} medicines</span>
    <strong>{percent:.0f}%</strong>
  </div>
  <div class="dss-readiness-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent:.0f}">
    <div class="dss-readiness-fill" style="width:{percent:.2f}%;"></div>
  </div>
  <div class="dss-readiness-note">{remaining:,} medicines still require data review or are missing forecast inputs.</div>
</section>
""",
        unsafe_allow_html=True,
    )


def executive_readiness_note(
    planning_medicines: int,
    optimizer_ready: int,
    review_items: int,
    expiry_medicines: int,
    readiness_score: float,
) -> None:
    """Display-only executive summary using the already loaded snapshot."""
    st.markdown(
        f"""
<section class="dss-executive-note">
  <div class="dss-executive-note-title">⚠&nbsp; Executive readiness summary</div>
  <div class="dss-executive-note-body">
    Planning is at {readiness_score:.0%} readiness. The majority of medicines
    ({optimizer_ready:,} of {planning_medicines:,}) are optimizer-ready, but
    {review_items:,} require review before the procurement plan can be finalized.
    Historical expiry coverage is available for {expiry_medicines:,} medicines.
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def _show_planning_action(action: str | None) -> None:
    """Open an existing evidence view from the overview's action cards."""
    st.session_state["planning_action_view"] = action


def planning_action_card_heading(
    title: str,
    description: str,
    *,
    evidence: bool = False,
) -> None:
    color = "#28855e" if evidence else "#986015"
    badge = "Evidence available" if evidence else "Action needed"
    icon = "governance" if evidence else "attention"
    st.markdown(
        '<div class="planning-action-top">'
        f'{icon_svg(icon, color, 23)}'
        f'<span class="planning-action-badge {"success" if evidence else ""}">'
        f'{icon_svg("check" if evidence else "attention", color, 15)}{badge}'
        '</span></div>'
        f'<div class="planning-action-title">{title}</div>'
        f'<div class="planning-action-description">{description}</div>',
        unsafe_allow_html=True,
    )


def render_planning_decision_status(snapshot: dict) -> None:
    """Reference-style action cards over the existing, read-only planning data."""
    stock_count = snapshot["Needs_Stock_Override"]
    cost_count = snapshot["Needs_Cost_Override"]
    review_count = stock_count + cost_count
    expiry_count = snapshot["With_Expiry_History"]

    with st.container(key="planning-decision-status"):
        with st.expander("Decision status", expanded=True):
            st.markdown(
                '<div class="planning-actions-intro">'
                '<span>Focus on these actions before approving the plan</span>'
                '<span>3 planning priorities</span></div>',
                unsafe_allow_html=True,
            )

            with st.container(key="planning-action-grid"):
                review_col, preparation_col, expiry_col = st.columns(
                    [1.15, 1, 1], gap="small"
                )
                with review_col:
                    with st.container(border=True, key="planning-action-card-review"):
                        planning_action_card_heading(
                            f"{review_count:,} medicines need review",
                            "Complete these two checks to improve the quality of your plan.",
                        )
                        with st.container(key="planning-action-review-rows"):
                            st.button(
                                f"**{stock_count:,}** need stock review",
                                key="planning-action-review-row-stock",
                                width="stretch",
                                on_click=_show_planning_action,
                                args=("stock",),
                            )
                            st.button(
                                f"**{cost_count:,}** need cost review",
                                key="planning-action-review-row-cost",
                                width="stretch",
                                on_click=_show_planning_action,
                                args=("cost",),
                            )
                        with st.container(key="planning-action-footer-review"):
                            st.button(
                                "View review worklist",
                                key="planning-action-open-worklist",
                                width="stretch",
                                on_click=_show_planning_action,
                                args=("worklist",),
                            )

                with preparation_col:
                    with st.container(border=True, key="planning-action-card-preparation"):
                        planning_action_card_heading(
                            "Complete preparation review",
                            "Verify planning inputs and confirm responsibilities before the final approval.",
                        )
                        st.markdown(
                            '<div class="planning-action-meta">'
                            '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" '
                            'stroke="#6b7b8c" stroke-width="1.3" aria-hidden="true">'
                            '<path d="M10 4C7 2 4 2 2 3v13c3-1 5-1 8 1 3-2 5-2 8-1V3c-2-1-5-1-8 1v13"/>'
                            '</svg><span>5 planning checks to review</span></div>',
                            unsafe_allow_html=True,
                        )
                        with st.container(key="planning-action-footer-preparation"):
                            st.button(
                                "Open checklist",
                                key="planning-action-open-checklist",
                                width="stretch",
                                on_click=_show_planning_action,
                                args=("checklist",),
                            )

                with expiry_col:
                    with st.container(border=True, key="planning-action-card-expiry"):
                        planning_action_card_heading(
                            "Review expiry evidence",
                            f"{expiry_count:,} medicines have linked history. "
                            "Use the evidence to support risk-aware purchasing.",
                            evidence=bool(expiry_count),
                        )
                        st.markdown(
                            '<div class="planning-action-meta">'
                            f'{icon_svg("expiry", "#6b7b8c", 18)}'
                            f'<span>Evidence period: {HISTORICAL_PERIOD_LABEL}</span></div>',
                            unsafe_allow_html=True,
                        )
                        with st.container(key="planning-action-footer-expiry"):
                            st.button(
                                "View expiry review",
                                key="planning-action-open-expiry",
                                width="stretch",
                                on_click=_activate_page,
                                args=("Expiry & Data Quality Review",),
                            )

            selected_action = st.session_state.get("planning_action_view")
            if selected_action:
                with st.container(key="planning-action-details"):
                    if selected_action == "checklist":
                        st.markdown("#### Preparation review checklist")
                        st.caption(
                            "Review the current evidence before approval. "
                            "These are review prompts; completion is not recorded here."
                        )
                        checks = pd.DataFrame(
                            [
                                ("Review current stock", f"{stock_count:,} stock review flags"),
                                ("Review planning costs", f"{cost_count:,} cost review flags"),
                                (
                                    "Confirm optimizer eligibility",
                                    f"{snapshot['Optimizer_Ready']:,} of "
                                    f"{snapshot['Planning_Medicines']:,} medicines are ready",
                                ),
                                (
                                    "Review forecast assumptions",
                                    f"{snapshot['Total_Point_Forecast']:,.0f} point-forecast units; "
                                    "review uncertainty in Forecast Planning",
                                ),
                                (
                                    "Review historical expiry evidence",
                                    f"{expiry_count:,} linked medicines; historical evidence only",
                                ),
                            ],
                            columns=["Planning check", "Current evidence"],
                        )
                        st.dataframe(checks, hide_index=True, width="stretch")
                    else:
                        titles = {
                            "stock": "Stock review worklist",
                            "cost": "Cost review worklist",
                            "worklist": "Stock and cost review worklist",
                        }
                        st.markdown(f"#### {titles.get(selected_action, 'Review worklist')}")
                        try:
                            inputs = get_procurement_medicine_inputs()
                            stock_flags = inputs["Needs_Current_Stock_Override"].fillna(False)
                            cost_flags = inputs["Needs_Planning_Cost_Override"].fillna(False)
                            mask = (
                                stock_flags if selected_action == "stock"
                                else cost_flags if selected_action == "cost"
                                else stock_flags | cost_flags
                            )
                            worklist = inputs.loc[mask].copy()
                            st.caption(
                                f"{len(worklist):,} matching medicines. "
                                "Stock and cost flags may overlap on the same medicine."
                            )
                            display_table(
                                worklist,
                                columns=[
                                    "Brand_Name", "Generic_Name", "Unit",
                                    "Planning_Stock_Qty", "Planning_Stock_Status",
                                    "Planning_Cost", "Planning_Cost_Status",
                                    "Planning_Input_Status",
                                ],
                            )
                        except Exception as error:
                            st.error("The review worklist could not be loaded.")
                            st.exception(error)
                    st.button(
                        "Close details",
                        key="planning-action-close-details",
                        on_click=_show_planning_action,
                        args=(None,),
                    )


def decision_status_card(
    value: str,
    title: str,
    message: str,
    severity: str,
) -> None:
    """Render a fixed-height dashboard status card; no business logic changes."""
    st.markdown(
        f"""
<section class="dss-status-card {severity}">
  <div class="dss-status-card-value">{value}</div>
  <div class="dss-status-card-title">{title}</div>
  <div class="dss-status-card-body">{message}</div>
</section>
""",
        unsafe_allow_html=True,
    )


def readiness_breakdown_table(
    planning_medicines: int,
    optimizer_ready: int,
    review_items: int,
    with_expiry_history: int,
    without_expiry_history: int,
    unresolved_identities: int,
) -> None:
    """Show live, overlapping evidence categories as a sortable table."""
    rows = [
        ("Optimizer-ready medicines", optimizer_ready, "Good"),
        ("Stock and cost review flags", review_items, "Needs review"),
        ("Unresolved identities", unresolved_identities, "Needs review"),
        ("Linked expiry history", with_expiry_history, "Evidence available"),
        ("No linked expiry history", without_expiry_history, "No expiry history"),
    ]
    data = pd.DataFrame(
        [
            {
                "Category": name,
                "Medicines": count,
                "Share_Pct": count / planning_medicines * 100 if planning_medicines else 0,
                "Status": status,
            }
            for name, count, status in rows
        ]
    )
    clone_table(
        data,
        key="planning-readiness-categories",
        labels={"Share_Pct": "Share"},
        page_size=8,
    )
    st.caption(
        "Review flags and identity issues may overlap with optimizer-ready records. "
        "These categories are not additive."
    )


COLUMN_LABELS = {
    "Brand_Name": "Brand Name",
    "Generic_Name": "Generic Name",
    "Unit": "Unit",
    "ABC_Class": "ABC Class",
    "ABC_AsOf": "ABC Class",
    "ABC_Eligibility": "ABC Eligibility",
    "Behavior_Segment": "Demand Pattern",
    "Behavior_Segment_AsOf": "Demand Pattern",
    "Latest_Annual_Issuance": "Last Year's Issuance",
    "Latest_Data_Quality_Status": "Data Quality",
    "Point_Forecast": "Forecast (Expected)",
    "P80": "P80 Scenario (Higher Buffer)",
    "P90": "P90 Scenario (Highest Buffer)",
    "Point_Forecast_Status": "Forecast Status",
    "Net_Requirement_Point": "Recommended Buy (Standard)",
    "Net_Requirement_P80": "Recommended Buy (Safer)",
    "Net_Requirement_P90": "Recommended Buy (Safest)",
    "Net_Requirement": "Recommended Buy",
    "Estimated_Procurement_Value_Point": "Estimated Cost (Standard)",
    "Estimated_Procurement_Value_P80": "Estimated Cost (Safer)",
    "Estimated_Procurement_Value_P90": "Estimated Cost (Safest)",
    "Planning_Input_Status": "Status",
    "Expiry_History_Coverage_Status": "History Coverage",
    "Expiry_History_Data_Quality_Status": "Data Quality",
    "Latest_Expiration_Date_Represented": "Most Recent Expiry Date on Record",
    "Expiry_DSS_Interpretation": "Interpretation",
    "Medicine_Count": "Medicine Count",
    "Historical_Total_Issuance": "Total Historical Issuance",
    "Issuance_Share_Pct": "Share of Total Issuance (%)",
    "Median_Persistence": "Median Persistence",
    "Median_Positive_CV": "Median Variability (CV)",
    "Known_ACV": "Known Annual Cost Value",
    "Known_ACV_Share": "Share of Total Value",
    "Medicine_Share_Pct": "Share of Medicines",
    "Requirement_Scenario": "Planning Scenario",
    "Budget_Percent": "Budget Level",
    "Scenario_Budget": "Budget Amount",
    "Required_Units": "Units Needed",
    "Allocated_Units": "Units Funded",
    "Unfunded_Requirement_Units": "Units Not Funded",
    "Allocated_Value": "Amount Allocated",
    "Budget_Utilization_Pct": "Budget Used (%)",
    "Item_Coverage_Ratio": "Coverage Ratio",
}


def display_table(df, columns=None, labels=None):
    """Render a DataFrame with selected columns and human-readable headers.
    Does not modify the underlying DataFrame or its values.
    """
    data = df[columns].copy() if columns else df.copy()
    data = data.rename(columns=(labels or COLUMN_LABELS))
    st.dataframe(
        data,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.Column(
                width="large" if data[column].dtype == "object" else "medium"
            )
            for column in data.columns
        },
    )


def display_raw_table(df):
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.Column(width="large")
            for column in df.columns
        },
    )


PROCUREMENT_EVIDENCE_LABELS = {
    "Validated Budget Scenarios": "Budget Scenario Validation",
    "Budget Scenario Summary": "Optimization Scenario Summary",
    "Budget Results by ABC": "ABC Allocation Summary",
    "Medicine Allocations": "Medicine Allocation Details",
    "Strict Priority Audit": "ABC Priority Validation",
    "MILP Governance": "Optimization Governance Record",
    "Solver Governance": "Solver Validation Record",
    "Planning Governance": "Planning Data Governance",
    "MILP Scenario Definition": "Scenario Definition",
    "Full-Funding Validation": "Full-Funding Validation Summary",
    "Full-Funding by ABC": "Full-Funding ABC Summary",
    "Full-Funding Allocations": "Full-Funding Allocation Details",
}

EXPIRY_EVIDENCE_LABELS = {
    "Medicine Expiry Summary": "Historical Expiry Summary",
    "Expiry Identity Crosswalk": "Medicine Identity Mapping",
    "Expiry Identity Review Queue": "Identity Review Queue",
    "Validated Expiry Records": "Validated Source Records",
    "Expiry Validation Summary": "Expiry Validation Summary",
    "Expiry DSS Governance": "Expiry Governance Record",
    "Expiry Source Governance": "Source Documentation",
    "Planning Review Queue": "Planning Review Queue",
    "Planning Review Summary": "Planning Review Summary",
}

RETROSPECTIVE_EVIDENCE_LABELS = {
    "Matched-Budget Summary": "Matched-Budget Summary",
    "Matched-Budget Trade-off": "Matched-Budget Trade-off Analysis",
    "Results by ABC": "Results by ABC Class",
    "Alternate-Optimum Ranges": "Alternate-Optimum Range Audit",
    "Policy Robustness": "Policy Robustness Analysis",
    "Full Repeatability Audit": "Full Repeatability Audit",
    "Repeatability Governance": "Repeatability Governance",
    "Alternate-Optimum Governance": "Alternate-Optimum Governance",
    "Matched-Budget Governance": "Matched-Budget Governance",
    "Retrospective Governance": "Retrospective Governance",
    "Simulation Governance": "Simulation Governance",
}

PROCUREMENT_TECHNICAL_COLUMNS = {
    "Medicine_Key",
    "Optimization_Version",
    "Run_ID",
    "Run_Mode",
    "Pyomo_Version",
    "HiGHS_Version",
    "Solver_Interface",
    "Theta_A",
    "Theta_B",
    "Theta_C",
    "Stage_1",
    "Stage_2",
    "Stage_3",
    "Stage_4",
    "Stage_5",
    "Stage_6",
    "Stage_7",
}


def _expiry_technical_column(column: str) -> bool:
    """Identify expiry provenance and implementation fields for display only."""
    technical_fragments = (
        "SHA256",
        "Source_",
        "Mapping_",
        "Description_Normalized",
        "Item_Prefix",
        "Medicine_Domain",
        "Pipeline_",
        "Build_",
        "Run_",
        "Version",
        "Row_Number",
        "Row_Count",
        "Flag_",
        "Computed_",
        "_Valid",
        "_Available",
        "_Computable",
        "_Claim",
        "_Supported",
        "_Removed",
        "_Accepted",
        "_Interpreted",
        "_Used",
        "Require_Review",
    )
    return column == "Item_ID" or any(
        fragment in column for fragment in technical_fragments
    )


def display_evidence_table(
    df,
    hidden_columns=None,
    expiry_view=False,
):
    """Render a decision-focused evidence view while preserving the raw table."""
    hidden = set(hidden_columns or ())
    if expiry_view:
        hidden.update(
            column
            for column in df.columns
            if _expiry_technical_column(column)
        )

    display_df = df.drop(columns=list(hidden), errors="ignore")
    display_df = display_df.rename(columns=COLUMN_LABELS)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.Column(
                width="large" if display_df[column].dtype == "object" else "medium"
            )
            for column in display_df.columns
        },
    )

    with st.expander("Show technical fields", expanded=False):
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                column: st.column_config.Column(width="large")
                for column in df.columns
            },
        )


def _severity_badge(severity: str) -> str:
    return {
        "ok": f'{icon_svg("check", "#2f7d5c")} OK',
        "attention": f'{icon_svg("attention", "#a66a1f")} Attention',
        "unavailable": f'{icon_svg("info", "#667085")} Not applicable',
    }.get(severity, f'{icon_svg("info", "#667085")} Not applicable')


def _toggle_question(page_key: str, question_key: str) -> None:
    state_key = f"expanded_q_{page_key}"
    st.session_state[state_key] = (
        None
        if st.session_state.get(state_key) == question_key
        else question_key
    )


def render_question_panel(
    page_key: str,
    questions,
    heading: str = "Decision questions",
) -> None:
    """Render live decision questions in the reference card presentation."""
    state_key = f"expanded_q_{page_key}"
    expanded_question = st.session_state.get(state_key)

    with st.container(key=f"question-panel-{page_key}"):
        with st.expander(heading, expanded=True):
            for question in questions:
                question_key = question["key"]
                severity = question["severity"]
                badge_class = (
                    "ok" if severity == "ok"
                    else "attention" if severity == "attention"
                    else "unavailable"
                )
                badge_text = question.get(
                    "badge_label",
                    {
                        "ok": "✓ On track",
                        "attention": "⚠ Needs attention",
                        "unavailable": "ⓘ Evidence view",
                    }.get(severity, "ⓘ Evidence view"),
                )
                with st.container(key=f"question-row-{page_key}-{question_key}"):
                    st.markdown(
                        '<div class="clone-question">'
                        '<div class="clone-question-icon">'
                        f'{icon_svg(question["icon"], "#1f6f78", 23)}'
                        '</div><div class="clone-question-content">'
                        '<div class="clone-question-line">'
                        f'<span class="clone-question-title">{html.escape(question["text"])}</span>'
                        f'<span class="dss-badge {badge_class}">'
                        f'{html.escape(badge_text)}</span></div>'
                        '<div class="clone-question-description">'
                        f'{html.escape(question.get("summary", question["interpretation"]))}'
                        '</div></div></div>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        question["text"],
                        key=f"question-action-{page_key}-{question_key}",
                        on_click=_toggle_question,
                        args=(page_key, question_key),
                    )

    if expanded_question:
        selected = next(
            (
                question
                for question in questions
                if question["key"] == expanded_question
            ),
            None,
        )
        if selected:
            st.markdown(selected["interpretation"])
            selected["render"]()


def workflow_step(
    number: int,
    title: str,
    description: str,
) -> None:


    st.markdown(
        f"""
<div style="
border-left:4px solid #1f6f78;
padding-left:15px;
margin-bottom:15px;
">


<b>
{number}. {title}
</b>


<br>


<span style="
opacity:0.75;
">
{description}
</span>


</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

def render_executive_dashboard() -> None:


    workspace_page_header(
        "Planning Overview",
        f"A clear picture of your medicine portfolio. A confident start to {PLANNING_YEAR}.",
    )


    try:

        snapshot = (
            get_executive_snapshot()
        )
        expiry_evidence = get_expiry_dss_summary()


    except Exception as error:

        st.error(
            "Executive dashboard data "
            "could not be loaded."
        )

        st.exception(error)

        return



    planning_medicines = (
        snapshot["Planning_Medicines"]
    )


    optimizer_ready = (
        snapshot["Optimizer_Ready"]
    )


    review_items = (
        snapshot["Needs_Stock_Override"]
        +
        snapshot["Needs_Cost_Override"]
    )


    readiness_rate = (

        optimizer_ready
        /
        planning_medicines

        if planning_medicines

        else 0
    )


    expiry_rate = (

        snapshot["With_Expiry_History"]
        /
        planning_medicines

        if planning_medicines

        else 0
    )



    readiness_score = (

        (
            readiness_rate * 0.45

            +

            expiry_rate * 0.25

            +

            (
                1
                -
                (
                    review_items
                    /
                    planning_medicines
                )
            )
            * 0.30
        )

        if planning_medicines

        else 0
    )


    readiness_score = max(
        0,
        min(
            readiness_score,
            1
        )
    )

    readiness_pct = readiness_rate * 100
    planning_question = {
        "key": "readiness",
        "icon": "governance",
        "text": f"Are we ready for {PLANNING_YEAR} planning?",
        "severity": (
            "ok"
            if readiness_pct >= 90
            else "attention"
        ),
        "interpretation": (
            f"{readiness_pct:.0f}% of medicines are ready for procurement "
            f"modeling ({optimizer_ready:,} of {planning_medicines:,}). "
            f"{planning_medicines - optimizer_ready:,} items require "
            "validation of planning inputs before they can be used in "
            "procurement decisions."
        ),
        "summary": (
            "Most medicines are ready. Resolve stock and cost reviews before "
            "finalizing the annual procurement plan."
        ),
        "render": lambda: decision_guardrail(
            "Executive Planning Summary",
            (
                f"The DSS evaluated {planning_medicines:,} medicines "
                f"for {PLANNING_YEAR} planning readiness. "
                f"{optimizer_ready:,} medicines are currently ready "
                "for procurement modeling. Remaining items require "
                "validation of planning inputs."
            ),
        ),
    }
    render_question_panel(
        "planning",
        [planning_question],
        heading="Decision question",
    )
    with clone_section("Executive readiness", "planning-executive"):
        st.markdown(
            '<div class="clone-intro"><span>Your planning portfolio, at a glance</span>'
            f'<span class="dss-workspace-badge info">ⓘ Evidence {HISTORICAL_PERIOD_LABEL}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="planning-metric-grid"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                workspace_metric_card(
                    "Planning medicines", f"{planning_medicines:,}",
                    "Medicines in the planning portfolio", "medicine", "ⓘ Portfolio",
                )
            with c2:
                workspace_metric_card(
                    "Optimizer-ready", f"{optimizer_ready:,}",
                    f"{readiness_rate:.0%} of the total portfolio", "governance",
                    "⚠ Review remaining" if readiness_rate < .9 else "✓ Ready",
                    "attention" if readiness_rate < .9 else "success",
                )
            with c3:
                workspace_metric_card(
                    f"{PLANNING_YEAR} point forecast",
                    f"{snapshot['Total_Point_Forecast']:,.0f}",
                    f"Units expected for the {PLANNING_YEAR} planning year",
                    "forecast", "ⓘ Point forecast",
                )
            with c4:
                workspace_metric_card(
                    "Linked expiry history", f"{snapshot['With_Expiry_History']:,}",
                    f"{expiry_rate:.0%} have historical expiry evidence",
                    "expiry", "ⓘ Evidence",
                )

        workspace_readiness_card(
            readiness_score, optimizer_ready, planning_medicines,
        )

        economic_profile_col, evidence_coverage_col = st.columns(2)
        with economic_profile_col:
            planning_donut_chart_card(
                "planning-economic-profile",
                "Economic profile distribution",
                "ABC classification by annual consumption value",
                ["Class A", "Class B", "Class C", "Unclassified"],
                [snapshot["ABC_A"], snapshot["ABC_B"], snapshot["ABC_C"],
                 snapshot["ABC_Unclassified_Cost"]],
                {"Class A": "#1f6f78", "Class B": "#d9a441",
                 "Class C": "#5b7b87", "Unclassified": "#b8c2cc"},
                "Class A medicines have the greatest budget impact.",
            )
        with evidence_coverage_col:
            planning_donut_chart_card(
                "planning-evidence-coverage", "Evidence coverage",
                "Historical expiry evidence across your portfolio",
                ["Linked history", "No linked history"],
                [snapshot["With_Expiry_History"], snapshot["Without_Expiry_History"]],
                {"Linked history": "#1f6f78", "No linked history": "#b8c2cc"},
                "History supports review; it does not predict future expiry.",
            )

    render_planning_decision_status(snapshot)

    with clone_section("Readiness breakdown by category", "planning-breakdown"):
        readiness_breakdown_table(
            planning_medicines,
            optimizer_ready,
            review_items,
            snapshot["With_Expiry_History"],
            snapshot["Without_Expiry_History"],
            int(expiry_evidence["Unresolved_Expiry_Identities"]),
        )


# ============================================================
# MEDICINES FOR REVIEW
# ============================================================

def _render_inventory_demand_profile_legacy() -> None:
    page_header(
        "Medicine Prioritization",
        "Medicine Prioritization",
        (
            "Explore historical demand behavior, "
            "economic classification, and medicine-level "
            "planning characteristics."
        ),
    )

    try:
        annual = get_annual_inventory_profile()
        abc = get_latest_abc_profile()
        behavior = get_demand_behavior_summary()
        matrix = get_latest_abc_behavior_matrix()
        explorer = get_inventory_demand_explorer()
    except Exception as error:
        st.error("Medicine review data could not be loaded.")
        st.exception(error)
        return

    abc_counts = {
        str(row["ABC_Class"]): int(row["Medicine_Count"])
        for _, row in abc.iterrows()
    }
    behavior_counts = {
        str(row["Behavior_Segment"]): int(row["Medicine_Count"])
        for _, row in behavior.iterrows()
    }
    sparse_count = behavior_counts.get("SPARSE", 0)
    nonmoving_count = behavior_counts.get("NON_MOVING", 0)
    variable_count = behavior_counts.get("PERSISTENT_VARIABLE", 0)
    stable_count = behavior_counts.get("PERSISTENT_STABLE", 0)

    render_question_panel(
        "medicine_review",
        [
            {
                "key": "medicine",
                "icon": "medicine",
                "text": "Which medicines are most important?",
                "severity": "ok",
                "interpretation": (
                    f"Medicines are grouped by economic importance. "
                    f"Class A ({abc_counts.get('A', 0):,} medicines) "
                    "represents the highest-value share of spend and should "
                    "be prioritized first if budget is constrained. Class B "
                    f"({abc_counts.get('B', 0):,}) is moderate priority. "
                    f"Class C ({abc_counts.get('C', 0):,}) is lowest priority "
                    "for reduction if funds are limited. This view reflects "
                    "economic value only, not clinical criticality; cross-check "
                    "critical or emergency medicines manually before "
                    "deprioritizing."
                ),
                "render": lambda: display_table(abc),
            },
            {
                "key": "demand",
                "icon": "forecast",
                "text": "Which medicines have unpredictable demand?",
                "severity": (
                    "attention"
                    if nonmoving_count + sparse_count > 0
                    else "ok"
                ),
                "interpretation": (
                    f"{stable_count:,} medicines show stable, predictable "
                    f"demand. {variable_count:,} show fluctuating demand and "
                    "should carry a safety buffer. "
                    f"{sparse_count:,} are rarely used; avoid overstocking. "
                    f"{nonmoving_count:,} show no recent demand and should be "
                    "reviewed before purchasing."
                ),
                "render": lambda: display_table(behavior),
            },
        ],
    )

    latest = annual.sort_values("Year").iloc[-1]

    section_title("Medicine portfolio snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Annual Issuance", f"{latest['Total_Issuance']:,.0f}")
    b.metric("Active Medicines", f"{int(latest['Active_Medicines']):,}")
    c.metric("Known ACV", f"₱{latest['Known_ACV']:,.2f}")
    d.metric("Medicine Records", f"{len(explorer):,}")

    st.divider()

    section_title(
        "Medicine Explorer",
        "Filter medicines by identity and planning characteristics.",
    )
    search = st.text_input(
        "Search medicine",
        placeholder="Brand name, generic name, or medicine key",
        key="medicine_review_search",
    )
    filtered = explorer.copy()

    if search.strip():
        value = search.strip()
        mask = (
            filtered["Brand_Name"].fillna("").str.contains(
                value, case=False, regex=False
            )
            | filtered["Generic_Name"].fillna("").str.contains(
                value, case=False, regex=False
            )
            | filtered["Medicine_Key"].fillna("").str.contains(
                value, case=False, regex=False
            )
        )
        filtered = filtered[mask]

    st.metric("Matching Medicines", f"{len(filtered):,}")
    display_table(
        filtered,
        columns=[
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "ABC_Class",
            "Behavior_Segment",
            "Latest_Annual_Issuance",
            "Latest_Data_Quality_Status",
        ],
    )

    with st.expander("Show all technical fields", expanded=False):
        display_raw_table(filtered)

    st.divider()

    with st.expander("Portfolio Insights", expanded=False):
        section_title("Classification profile")
        x, y = st.columns(2)
        with x:
            display_table(abc)
        with y:
            display_table(behavior)

        section_title("Historical demand profile")
        left, right = st.columns(2)
        with left:
            fig = px.line(
                annual,
                x="Year",
                y="Total_Issuance",
                markers=True,
                title="Annual issuance trend",
            )
            style_chart(fig)
            st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.bar(
                annual,
                x="Year",
                y="Active_Medicines",
                title="Active medicine trend",
                color_discrete_sequence=CHART_COLORS,
            )
            style_chart(fig)
            st.plotly_chart(fig, width="stretch")

        section_title("ABC and Demand Behavior Matrix")
        matrix_chart = (
            matrix.drop(columns=["Year"], errors="ignore")
            .melt(
                id_vars=["ABC_Class"],
                var_name="Behavior_Segment",
                value_name="Medicine_Count",
            )
        )
        fig = px.density_heatmap(
            matrix_chart,
            x="ABC_Class",
            y="Behavior_Segment",
            z="Medicine_Count",
            text_auto=True,
            color_continuous_scale=["#f5f8f8", "#1f6f78"],
        )
        style_chart(fig)
        st.plotly_chart(fig, width="stretch")

    decision_guardrail(
        "Medicine Review Interpretation",
        (
            "Medicine classifications describe historical consumption "
            "behavior and economic contribution. They support planning "
            "prioritization but do not represent clinical importance."
        ),
    )

# ============================================================
# MEDICINE PRIORITIZATION (REFERENCE UI)
# ============================================================

def render_medicine_explorer(explorer: pd.DataFrame) -> None:
    """Search and filter the connected medicine records for display only."""
    with clone_section("Medicine explorer", "medicine-explorer"):
        st.markdown('<span id="medicine-explorer" class="clone-anchor"></span>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="clone-intro"><span>ⓘ Start with Class A medicines, '
            'then review variable or sparse demand.</span></div>',
            unsafe_allow_html=True,
        )
        search_col, count_col = st.columns([5, 1.2], vertical_alignment="center")
        with search_col:
            search = st.text_input(
                "Search medicine",
                placeholder="Search brand, generic name, or medicine key…",
                key="medicine_review_search",
                label_visibility="collapsed",
            )
        with st.container(key="clone-medicine-filters"):
            pattern_col, class_col, quality_col = st.columns(3)
            with pattern_col:
                demand_filter = st.selectbox(
                    "Demand pattern",
                    ["All demand patterns"] + sorted(
                        explorer["Behavior_Segment"].dropna().astype(str).unique().tolist()
                    ),
                    key="medicine_review_demand_filter",
                    format_func=lambda value: value.replace("_", " ").capitalize(),
                )
            with class_col:
                class_filter = st.selectbox(
                    "Economic class",
                    ["All classes"] + sorted(
                        explorer["ABC_Class"].dropna().astype(str).unique().tolist()
                    ),
                    key="medicine_review_abc_filter",
                )
            with quality_col:
                quality_filter = st.selectbox(
                    "Data quality",
                    ["All quality statuses"] + sorted(
                        explorer["Latest_Data_Quality_Status"].dropna().astype(str).unique().tolist()
                    ),
                    key="medicine_review_quality_filter",
                    format_func=lambda value: value.replace("_", " ").capitalize(),
                )

        filtered = explorer.copy()
        if search.strip():
            value = search.strip()
            mask = pd.Series(False, index=filtered.index)
            for column in ("Brand_Name", "Generic_Name", "Medicine_Key"):
                if column in filtered:
                    mask |= filtered[column].fillna("").astype(str).str.contains(
                        value, case=False, regex=False,
                    )
            filtered = filtered[mask]
        if demand_filter != "All demand patterns":
            filtered = filtered[filtered["Behavior_Segment"] == demand_filter]
        if class_filter != "All classes":
            filtered = filtered[filtered["ABC_Class"] == class_filter]
        if quality_filter != "All quality statuses":
            filtered = filtered[
                filtered["Latest_Data_Quality_Status"] == quality_filter
            ]
        with count_col:
            st.markdown(
                f'<div class="dss-priority-explorer-count"><strong>{len(filtered):,}'
                '</strong> medicines</div>',
                unsafe_allow_html=True,
            )
        clone_table(
            filtered[
                ["Brand_Name", "Generic_Name", "Unit", "ABC_Class",
                 "Behavior_Segment", "Latest_Annual_Issuance",
                 "Latest_Data_Quality_Status"]
            ],
            key="medicine-prioritization-records",
            labels=COLUMN_LABELS,
            page_size=8,
        )
        st.caption(
            f"{len(filtered):,} matching live records. Portfolio total: "
            f"{len(explorer):,} medicines. Filters affect this table only."
        )
        with st.expander("Show all technical fields", expanded=False):
            display_raw_table(filtered)


def render_inventory_demand_profile() -> None:
    """Live Medicine Prioritization view; data and graph calculations stay intact."""
    workspace_page_header(
        "Medicine Prioritization",
        (
            "Focus on the medicines that matter most, and understand where "
            "demand is uncertain."
        ),
    )

    try:
        annual = get_annual_inventory_profile()
        abc = get_latest_abc_profile()
        behavior = get_demand_behavior_summary()
        matrix = get_latest_abc_behavior_matrix()
        explorer = get_inventory_demand_explorer()
    except Exception as error:
        st.error("Medicine review data could not be loaded.")
        st.exception(error)
        return

    abc_counts = {
        str(row["ABC_Class"]): int(row["Medicine_Count"])
        for _, row in abc.iterrows()
    }
    behavior_counts = {
        str(row["Behavior_Segment"]): int(row["Medicine_Count"])
        for _, row in behavior.iterrows()
    }
    sparse_count = behavior_counts.get("SPARSE", 0)
    nonmoving_count = behavior_counts.get("NON_MOVING", 0)
    variable_count = behavior_counts.get("PERSISTENT_VARIABLE", 0)
    stable_count = behavior_counts.get("PERSISTENT_STABLE", 0)

    render_question_panel(
        "medicine_review",
        [
            {
                "key": "medicine",
                "icon": "medicine",
                "text": "Which medicines are most important?",
                "severity": "unavailable",
                "interpretation": (
                    f"Medicines are grouped by economic importance. "
                    f"Class A ({abc_counts.get('A', 0):,} medicines) "
                    "represents the highest-value share of spend and should "
                    "be prioritized first if budget is constrained. Class B "
                    f"({abc_counts.get('B', 0):,}) is moderate priority. "
                    f"Class C ({abc_counts.get('C', 0):,}) is lowest priority "
                    "for reduction if funds are limited. This view reflects "
                    "economic value only, not clinical criticality; cross-check "
                    "critical or emergency medicines manually before "
                    "deprioritizing."
                ),
                "summary": (
                    "Class A medicines carry the greatest annual consumption "
                    "value. Review them first."
                ),
                "render": lambda: display_table(abc),
            },
            {
                "key": "demand",
                "icon": "forecast",
                "text": "Which medicines have unpredictable demand?",
                "severity": (
                    "attention"
                    if nonmoving_count + sparse_count > 0
                    else "ok"
                ),
                "interpretation": (
                    f"{stable_count:,} medicines show stable, predictable "
                    f"demand. {variable_count:,} show fluctuating demand and "
                    "should carry a safety buffer. "
                    f"{sparse_count:,} are rarely used; avoid overstocking. "
                    f"{nonmoving_count:,} show no recent demand and should be "
                    "reviewed before purchasing."
                ),
                "summary": (
                    "Variable and sparse demand need closer attention to "
                    "avoid stockouts or excess inventory."
                ),
                "render": lambda: display_table(behavior),
            },
        ],
        heading="Decision questions",
    )

    latest = annual.sort_values("Year").iloc[-1]
    with st.container(key="priority-metric-grid"):
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        with metric_1:
            workspace_metric_card(
                "Annual issuance",
                f"{latest['Total_Issuance']:,.0f}",
                "Units issued across the portfolio",
                "forecast", "ⓘ Planning data",
                footer_label="View details", footer_href="#historical-demand",
            )
        with metric_2:
            workspace_metric_card(
                "Active medicines",
                f"{int(latest['Active_Medicines']):,}",
                "Medicines issued in the latest year",
                "medicine", "ⓘ Planning data",
                footer_label="View details", footer_href="#historical-demand",
            )
        with metric_3:
            workspace_metric_card(
                "Known ACV",
                f"&#8369;{latest['Known_ACV']:,.2f}",
                "Annual consumption value",
                "forecast", "ⓘ Planning data",
                value_tone="teal",
                footer_label="View details", footer_href="#classification-profile",
            )
        with metric_4:
            workspace_metric_card(
                "Medicine records",
                f"{len(explorer):,}",
                "Records in the planning portfolio",
                "overview", "ⓘ Planning data",
                footer_label="View details", footer_href="#medicine-explorer",
            )

    st.markdown(
        '<div class="clone-jump-links"><strong>◇ &nbsp; Your portfolio, in one place</strong>'
        '<div><a href="#medicine-explorer">⌕ &nbsp; Medicine explorer</a>'
        '<a href="#classification-profile">⌁ &nbsp; Classification profile</a>'
        '<a href="#demand-matrix">▦ &nbsp; Demand matrix</a></div></div>',
        unsafe_allow_html=True,
    )
    render_medicine_explorer(explorer)

    with clone_section("Classification profile", "medicine-classification"):
        st.markdown('<span id="classification-profile" class="clone-anchor"></span>',
                    unsafe_allow_html=True)
        abc_chart_col, behavior_chart_col = st.columns(2)

        abc_chart_data = abc.copy()
        abc_chart_data["Display_Class"] = abc_chart_data["ABC_Class"].replace(
            {
                "A": "Class A",
                "B": "Class B",
                "C": "Class C",
                "UNCLASSIFIED_COST": "Unclassified",
            }
        )
        with abc_chart_col:
            st.markdown(
                '<div class="dss-priority-chart-title">By economic class</div>',
                unsafe_allow_html=True,
            )
            abc_figure = px.bar(
                abc_chart_data,
                x="Medicine_Count",
                y="Display_Class",
                orientation="h",
                text="Medicine_Count",
                color="Display_Class",
                color_discrete_map={
                    "Class A": "#1d7b84",
                    "Class B": "#dda63b",
                    "Class C": "#587987",
                    "Unclassified": "#b7c4cf",
                },
            )
            abc_figure.update_traces(texttemplate="%{text:,}", textposition="outside")
            abc_figure.update_layout(
                height=300,
                margin={"l": 4, "r": 40, "t": 10, "b": 24},
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=False,
                font={"family": "Inter, Segoe UI, sans-serif", "color": "#102a59"},
            )
            abc_figure.update_xaxes(showgrid=True, gridcolor="#e3eaee", title=None)
            abc_figure.update_yaxes(
                categoryorder="array",
                categoryarray=["Unclassified", "Class C", "Class B", "Class A"],
                title=None,
            )
            st.plotly_chart(
                abc_figure,
                width="stretch",
                config={"displayModeBar": False},
            )

        behavior_chart_data = behavior.copy()
        behavior_chart_data["Display_Behavior"] = behavior_chart_data[
            "Behavior_Segment"
        ].replace(
            {
                "PERSISTENT_STABLE": "Persistent stable",
                "PERSISTENT_VARIABLE": "Persistent variable",
                "SPARSE": "Sparse",
                "NON_MOVING": "Non-moving",
            }
        )
        with behavior_chart_col:
            st.markdown(
                '<div class="dss-priority-chart-title">By demand behavior</div>',
                unsafe_allow_html=True,
            )
            behavior_figure = px.bar(
                behavior_chart_data,
                x="Medicine_Count",
                y="Display_Behavior",
                orientation="h",
                text="Medicine_Count",
                color="Display_Behavior",
                color_discrete_map={
                    "Persistent stable": "#1d7b84",
                    "Persistent variable": "#587987",
                    "Sparse": "#dda63b",
                    "Non-moving": "#b7c4cf",
                },
            )
            behavior_figure.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
            )
            behavior_figure.update_layout(
                height=300,
                margin={"l": 4, "r": 40, "t": 10, "b": 24},
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=False,
                font={"family": "Inter, Segoe UI, sans-serif", "color": "#102a59"},
            )
            behavior_figure.update_xaxes(showgrid=True, gridcolor="#e3eaee", title=None)
            behavior_figure.update_yaxes(
                categoryorder="array",
                categoryarray=[
                    "Non-moving",
                    "Sparse",
                    "Persistent variable",
                    "Persistent stable",
                ],
                title=None,
            )
            st.plotly_chart(
                behavior_figure,
                width="stretch",
                config={"displayModeBar": False},
            )

        with st.expander("View classification details", expanded=False):
            detail_left, detail_right = st.columns(2)
            with detail_left:
                display_table(abc)
            with detail_right:
                display_table(behavior)

    with clone_section("Historical demand profile", "medicine-historical"):
        st.markdown('<span id="historical-demand" class="clone-anchor"></span>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="clone-evidence-pair">'
            '<div class="clone-evidence-callout"><strong>ⓘ &nbsp; Annual issuance</strong>'
            f'<span>{latest["Total_Issuance"]:,.0f} units were issued in '
            f'{int(latest["Year"])}. The chart below shows the available '
            'year-by-year evidence.</span></div>'
            '<div class="clone-evidence-callout"><strong>ⓘ &nbsp; Active medicine trend</strong>'
            f'<span>{int(latest["Active_Medicines"]):,} medicines were active '
            'in the latest year. Use the chart below to review the historical series.'
            '</span></div></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            # Existing graph retained exactly: only its surrounding card changed.
            fig = px.line(
                annual,
                x="Year",
                y="Total_Issuance",
                markers=True,
                title="Annual issuance trend",
            )
            style_chart(fig)
            st.plotly_chart(fig, width="stretch")
        with right:
            # Existing graph retained exactly: only its surrounding card changed.
            fig = px.bar(
                annual,
                x="Year",
                y="Active_Medicines",
                title="Active medicine trend",
                color_discrete_sequence=CHART_COLORS,
            )
            style_chart(fig)
            st.plotly_chart(fig, width="stretch")

    with clone_section("Economic class × demand behavior", "medicine-matrix"):
        st.markdown('<span id="demand-matrix" class="clone-anchor"></span>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="clone-intro">Count and share of current medicines by '
            'ABC economic class and demand behavior.</div>',
            unsafe_allow_html=True,
        )
        classes = [
            ("A", "Class A"), ("B", "Class B"),
            ("C", "Class C"), ("UNCLASSIFIED_COST", "Unclassified"),
        ]
        patterns = [
            ("PERSISTENT_STABLE", "Persistent stable"),
            ("PERSISTENT_VARIABLE", "Persistent variable"),
            ("SPARSE", "Sparse"), ("NON_MOVING", "Non-moving"),
        ]
        matrix_values = {
            (str(row["ABC_Class"]), pattern): int(row[pattern])
            for _, row in matrix.iterrows()
            for pattern, _ in patterns
        }
        maximum = max(matrix_values.values(), default=0)
        matrix_parts = ['<div class="clone-matrix" role="table" '
                        'aria-label="ABC class and demand behavior matrix">',
                        '<div class="clone-matrix-row clone-matrix-head" role="row">'
                        '<span role="columnheader"></span>']
        matrix_parts.extend(
            f'<span role="columnheader">{html.escape(label)}</span>'
            for _, label in classes
        )
        matrix_parts.append('</div>')
        for pattern, pattern_label in patterns:
            matrix_parts.append(
                '<div class="clone-matrix-row" role="row">'
                f'<span role="rowheader">{html.escape(pattern_label)}</span>'
            )
            for category, _ in classes:
                count = matrix_values.get((category, pattern), 0)
                share = count / len(explorer) * 100 if len(explorer) else 0.0
                tone = (
                    "strong" if maximum and count >= maximum * .3
                    else "warm" if maximum and count >= maximum * .08
                    else "light" if count else "empty"
                )
                matrix_parts.append(
                    f'<div class="clone-matrix-cell {tone}" role="cell">'
                    f'<strong>{count:,}</strong><span>{share:.1f}% of total</span>'
                    '</div>'
                )
            matrix_parts.append('</div>')
        matrix_parts.append('</div>')
        st.markdown(''.join(matrix_parts), unsafe_allow_html=True)
        st.markdown(
            '<div class="clone-plot-label">Interactive source heatmap</div>',
            unsafe_allow_html=True,
        )
        # Existing graph retained exactly: the data frame transformation and
        # heatmap settings are unchanged from the prior implementation.
        matrix_chart = (
            matrix.drop(columns=["Year"], errors="ignore")
            .melt(
                id_vars=["ABC_Class"],
                var_name="Behavior_Segment",
                value_name="Medicine_Count",
            )
        )
        fig = px.density_heatmap(
            matrix_chart,
            x="ABC_Class",
            y="Behavior_Segment",
            z="Medicine_Count",
            text_auto=True,
            color_continuous_scale=["#f5f8f8", "#1f6f78"],
        )
        style_chart(fig)
        st.plotly_chart(fig, width="stretch")

    with clone_section("Forecast & inventory coverage", "medicine-coverage"):
        forecast_col, stock_col = st.columns(2)
        with forecast_col:
            with st.container(key="clone-medicine-forecast-action"):
                st.markdown(
                    '<div class="clone-action-message"><strong>ⓘ &nbsp; Review forecast demand</strong>'
                    '<span>Compare forecast scenarios with historical issuance and '
                    'demand behavior before finalizing requirements.</span></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Review forecast demand →",
                    key="medicine-open-forecast",
                    on_click=_activate_page,
                    args=("Forecast Planning",),
                )
        with stock_col:
            with st.container(key="clone-medicine-stock-action"):
                st.markdown(
                    '<div class="clone-action-message"><strong>⚠ &nbsp; Verify stock inputs</strong>'
                    '<span>Review non-moving medicines and confirm planning stock '
                    'before interpreting budget coverage as a purchase recommendation.'
                    '</span></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Review procurement coverage →",
                    key="medicine-open-procurement",
                    on_click=_activate_page,
                    args=("Procurement Planning",),
                )

    decision_guardrail(
        "Medicine Review Interpretation",
        (
            "Medicine classifications describe historical consumption "
            "behavior and economic contribution. They support planning "
            "prioritization but do not represent clinical importance."
        ),
    )


# ============================================================
# FORECASTING & USAGE INSIGHTS
# ============================================================

def render_forecasting_uncertainty() -> None:
    workspace_page_header(
        "Forecast Planning",
        f"Prepare for {PLANNING_YEAR} with a clear baseline and practical demand buffers.",
    )

    try:
        summary = get_forecast_summary()
        profile = get_forecast_profile()
        by_behavior = get_forecast_by_behavior()
        by_abc = get_forecast_by_abc()
    except Exception as error:
        st.error("Forecast data could not be loaded.")
        st.exception(error)
        return

    forecast_display_columns = [
        "Brand_Name",
        "Generic_Name",
        "Unit",
        "ABC_AsOf",
        "Behavior_Segment_AsOf",
        "Point_Forecast",
        "P80",
        "P90",
        "Point_Forecast_Status",
    ]
    render_question_panel(
        "forecast",
        [
            {
                "key": "prepare",
                "icon": "forecast",
                "text": f"What medicines do we need to prepare for {PLANNING_YEAR}?",
                "severity": "unavailable",
                "badge_label": "ⓘ Planning scenarios",
                "summary": (
                    "Use the point forecast as a baseline, then compare "
                    "higher-buffer scenarios."
                ),
                "interpretation": (
                    "The table below lists forecasted demand for each "
                    "medicine. Prioritize items that are both high forecast "
                    "volume and Class A; these are the top procurement "
                    "candidates."
                ),
                "render": lambda: display_table(
                    profile,
                    columns=forecast_display_columns,
                ),
            },
            {
                "key": "accuracy",
                "icon": "validation",
                "text": "How accurate were previous forecasts?",
                "severity": "unavailable",
                "badge_label": "◷ Not available yet",
                "summary": (
                    "A validated comparison of historical forecasts and actual "
                    "issuance is not available in this view."
                ),
                "interpretation": (
                    "Historical forecast-accuracy comparison is not yet "
                    "available in this system."
                ),
                "render": lambda: None,
            },
        ],
    )

    with st.container(key="clone-forecast-kpis"):
        a, b, c, d = st.columns(4)
        with a:
            workspace_metric_card(
                "Point forecast", f"{summary['Point_Total']:,.0f}",
                "The baseline estimate of units needed for the planning year.",
                "forecast", "ⓘ Baseline", value_tone="teal",
            )
        with b:
            workspace_metric_card(
                "P80 scenario", f"{summary['P80_Total']:,.0f}",
                "A higher demand buffer: model 80th percentile.",
                "forecast", "ⓘ Higher buffer",
            )
        with c:
            workspace_metric_card(
                "P90 scenario", f"{summary['P90_Total']:,.0f}",
                "The highest buffer: model 90th percentile.",
                "forecast", "⚠ Highest buffer", badge_tone="attention",
            )
        with d:
            workspace_metric_card(
                "Quantile adjustments",
                f"{summary['Quantile_Adjusted_Medicines']:,}",
                "Records adjusted to keep scenarios in a consistent order.",
                "forecast", "ⓘ Planning data",
            )

    with clone_section("Medicine forecast explorer", "forecast-explorer"):
        search_col, count_col = st.columns(
            [7, 1.2], vertical_alignment="center",
        )
        with search_col:
            search = st.text_input(
                "Search forecasts", key="clone-forecast-search",
                placeholder="Search brand, generic name, or medicine key…",
                label_visibility="collapsed",
            )
        with st.container(key="clone-forecast-filters"):
            pattern_col, class_col, status_col = st.columns(3)
            with pattern_col:
                behavior_filter = st.selectbox(
                    "Demand pattern",
                    ["All demand patterns"] + sorted(
                        profile["Behavior_Segment_AsOf"].dropna().unique().tolist()
                    ),
                    key="clone-forecast-behavior",
                    format_func=lambda value: value.replace("_", " ").capitalize(),
                )
            with class_col:
                class_filter = st.selectbox(
                    "Economic class",
                    ["All classes"] + sorted(profile["ABC_AsOf"].dropna().unique().tolist()),
                    key="clone-forecast-class",
                )
            with status_col:
                status_filter = st.selectbox(
                    "Forecast status",
                    ["All forecast statuses"] + sorted(
                        profile["Point_Forecast_Status"].dropna().unique().tolist()
                    ),
                    key="clone-forecast-status",
                    format_func=lambda value: value.replace("_", " ").capitalize(),
                )
        filtered = profile.copy()
        if search.strip():
            mask = pd.Series(False, index=filtered.index)
            for column in ("Brand_Name", "Generic_Name", "Medicine_Key"):
                if column in filtered:
                    mask |= filtered[column].fillna("").astype(str).str.contains(
                        search.strip(), case=False, regex=False,
                    )
            filtered = filtered.loc[mask]
        if behavior_filter != "All demand patterns":
            filtered = filtered.loc[filtered["Behavior_Segment_AsOf"] == behavior_filter]
        if class_filter != "All classes":
            filtered = filtered.loc[filtered["ABC_AsOf"] == class_filter]
        if status_filter != "All forecast statuses":
            filtered = filtered.loc[filtered["Point_Forecast_Status"] == status_filter]
        with count_col:
            st.markdown(f"**{len(filtered):,}** medicines")
        clone_table(
            filtered[forecast_display_columns], key="forecast-explorer-table",
            labels={
                **COLUMN_LABELS, "Point_Forecast": "Point forecast",
                "P80": "P80", "P90": "P90",
            },
        )
        st.caption(
            "Forecasts use the connected analytical database. "
            "Filtering changes this display only, not the forecast models."
        )
        with st.expander("Show all technical fields", expanded=False):
            display_raw_table(filtered)

    behavior_col, abc_col = st.columns(2)
    scenario_colors = {
        "Point_Forecast": "#1F6F78", "P80": "#5B7B87", "P90": "#B8741A",
    }
    for container, title, section_key, source, category in (
        (behavior_col, "Forecast by demand behavior", "forecast-behavior", by_behavior, "Behavior_Segment"),
        (abc_col, "Forecast by economic class", "forecast-abc", by_abc, "ABC_Class"),
    ):
        with container, clone_section(title, section_key):
            chart = source[[category, "Point_Forecast", "P80", "P90"]].melt(
                id_vars=[category], var_name="Scenario", value_name="Units",
            )
            fig = px.bar(
                chart, x=category, y="Units", color="Scenario", barmode="group",
                color_discrete_map=scenario_colors,
            )
            style_chart(fig)
            fig.update_layout(
                height=410, margin=dict(l=0, r=8, t=62, b=16),
                font=dict(family="Inter, Segoe UI, sans-serif", color="#5B6B7B", size=14),
                legend=dict(title=None, orientation="h", y=1.18, x=0, font=dict(size=13)),
                xaxis_title=None, yaxis_title="Units", bargap=0.27, bargroupgap=0.08,
            )
            fig.for_each_trace(lambda trace: trace.update(name={
                "Point_Forecast": "Point forecast", "P80": "P80 · higher buffer",
                "P90": "P90 · highest buffer",
            }.get(trace.name, trace.name)))
            categories = source[category].tolist()
            fig.update_xaxes(
                tickmode="array", tickvals=categories,
                ticktext=[
                    (f"Class {value}" if value in ("A", "B", "C")
                     else str(value).replace("_", " ").capitalize())
                    for value in categories
                ],
                showgrid=False,
            )
            fig.update_yaxes(gridcolor="#E3E8EB", zeroline=False)
            st.plotly_chart(fig, width="stretch", key=f"clone-chart-{section_key}")

    decision_guardrail(
        "Forecast interpretation",
        (
            "Point is the baseline estimate. P80 and P90 are higher model demand "
            "percentiles, not guaranteed service levels. Review uncertainty and "
            "clinical requirements with authorized pharmacy staff before choosing "
            "a procurement scenario."
        ),
    )



# ============================================================
# PROCUREMENT PLANNING
# ============================================================

def render_procurement_decision_support() -> None:
    workspace_page_header(
        "Procurement Planning",
        "Compare budget coverage and review what can be funded before placing orders.",
    )

    try:
        summary = get_procurement_readiness_summary()
        medicines = get_procurement_medicine_inputs()
        options = get_procurement_optimizer_options()
    except Exception as error:
        st.error("Procurement data could not be loaded.")
        st.exception(error)
        return

    procurement_display_columns = [
        "Brand_Name", "Generic_Name", "Unit", "ABC_AsOf",
        "Behavior_Segment_AsOf", "Net_Requirement_Point",
        "Net_Requirement_P80", "Net_Requirement_P90",
        "Estimated_Procurement_Value_Point",
        "Estimated_Procurement_Value_P80",
        "Estimated_Procurement_Value_P90", "Planning_Input_Status",
    ]
    optimizer_overview_columns = [
        "Requirement_Scenario", "Budget_Percent", "Scenario_Budget",
        "Required_Units", "Allocated_Units", "Unfunded_Requirement_Units",
        "Allocated_Value", "Budget_Utilization_Pct",
    ]
    optimizer_allocation_columns = [
        "Brand_Name", "Generic_Name", "Unit", "ABC_AsOf",
        "Net_Requirement", "Allocated_Units",
        "Unfunded_Requirement_Units", "Allocated_Value", "Item_Coverage_Ratio",
    ]
    policies = options["Requirement_Scenario"].drop_duplicates().tolist()
    budgets = sorted(options["Budget_Percent"].drop_duplicates().tolist())
    if not policies or not budgets:
        st.info("Stored procurement scenarios are not available for this planning cycle.")
        return

    # The overview selects existing optimizer outputs; it never interpolates or
    # reruns the model. All original policy controls remain in Detailed workspace.
    reference_policy = next(
        (policy for policy in policies if str(policy).upper() == "POINT"),
        policies[0],
    )
    point_budgets = sorted(
        float(value)
        for value in options.loc[
            options["Requirement_Scenario"] == reference_policy,
            "Budget_Percent",
        ].drop_duplicates()
    )
    lowest_budget, full_budget = point_budgets[0], point_budgets[-1]
    budget_key = "procurement_clone_budget"
    if st.session_state.get(budget_key) not in point_budgets:
        st.session_state[budget_key] = int(full_budget)
    overview_budget = float(st.session_state[budget_key])
    reference_budget_results = {
        value: get_procurement_optimizer_result(reference_policy, value)
        for value in point_budgets
    }
    full_result = reference_budget_results[full_budget]
    lowest_result = reference_budget_results[lowest_budget]
    selected_result = reference_budget_results[overview_budget]
    full_overview = full_result["overview"].iloc[0]
    selected_overview = selected_result["overview"].iloc[0]
    full_required_units = float(full_overview["Required_Units"])
    full_allocated_units = float(full_overview["Allocated_Units"])
    fully_funded_pct = (
        full_allocated_units / full_required_units * 100
        if full_required_units else 100
    )
    required_units = float(selected_overview["Required_Units"])
    allocated_units = float(selected_overview["Allocated_Units"])
    unfunded_units = float(selected_overview["Unfunded_Requirement_Units"])
    selected_budget_value = float(selected_overview["Scenario_Budget"])
    selected_has_gap = unfunded_units > 0
    low_unfunded = lowest_result["allocations"]["Unfunded_Requirement_Units"]
    unfunded_medicine_count = int((low_unfunded > 0).sum())
    lowest_unfunded_units = float(low_unfunded.sum())

    def format_peso(value: float) -> str:
        return f"₱{float(value):,.2f}"

    def render_optimizer_question_result(result):
        display_table(result["overview"], columns=optimizer_overview_columns)
        display_table(
            result["allocations"],
            columns=[
                column for column in optimizer_allocation_columns
                if column in result["allocations"].columns
            ],
        )

    render_question_panel(
        "procurement",
        [
            {
                "key": "buy",
                "icon": "procurement",
                "text": "How much medicine should we buy?",
                "severity": "ok" if fully_funded_pct >= 99.99 else "attention",
                "badge_label": (
                    "✓ On track" if fully_funded_pct >= 99.99
                    else "⚠ Needs attention"
                ),
                "summary": (
                    "Review forecast requirements alongside verified planning "
                    "stock, unit costs, and the available budget."
                ),
                "interpretation": (
                    "Purchase requirements account for the forecast and the "
                    "planning stock inputs. Point, P80, and P90 are separate "
                    "planning scenarios, not guaranteed service levels."
                ),
                "render": lambda: display_table(
                    medicines, columns=procurement_display_columns,
                ),
            },
            {
                "key": "budget",
                "icon": "forecast",
                "text": "Can our budget cover the medicines?",
                "severity": "attention" if selected_has_gap else "ok",
                "badge_label": (
                    "⚠ Needs attention" if selected_has_gap else "✓ On track"
                ),
                "summary": (
                    f"The {overview_budget:.0f}% point scenario has a budget of "
                    f"{format_peso(selected_budget_value)}."
                ),
                "interpretation": (
                    f"At {overview_budget:.0f}% budget, {allocated_units:,.0f} "
                    f"of {required_units:,.0f} required units are funded. "
                    "The table shows the stored optimizer result for this scenario."
                ),
                "render": lambda: render_optimizer_question_result(selected_result),
            },
            {
                "key": "unfunded",
                "icon": "attention",
                "text": "Which medicines are unfunded if the budget is limited?",
                "severity": "attention" if lowest_unfunded_units > 0 else "ok",
                "badge_label": (
                    "⚠ Needs attention" if lowest_unfunded_units > 0 else "✓ On track"
                ),
                "summary": (
                    f"At {lowest_budget:.0f}% budget, "
                    f"{unfunded_medicine_count:,} medicines have unfunded "
                    "requirements. Use the allocation explorer to review them."
                ),
                "interpretation": (
                    "Allocation follows the governed ABC economic-priority "
                    "optimization. ABC class represents annual consumption value, "
                    "not clinical importance. Review the actual funding gaps below."
                ),
                "render": lambda: display_table(
                    selected_result["allocations"],
                    columns=[
                        column for column in optimizer_allocation_columns
                        if column in selected_result["allocations"].columns
                    ],
                ),
            },
        ],
    )

    coverage_description = (
        "all modeled eligible requirements are covered"
        if fully_funded_pct >= 99.99
        else f"{fully_funded_pct:.1f}% of modeled eligible requirements are covered"
    )
    st.markdown(
        f"""
<div class="dss-clone-callout">
  <span class="dss-clone-callout-icon">{icon_svg("info", "#1f6f78", 22)}</span>
  <div><strong>Budget impact at a glance</strong>
  <p>At {full_budget:.0f}% budget, {coverage_description}.
  At {lowest_budget:.0f}% budget, {unfunded_medicine_count:,} medicines have
  unfunded requirements ({lowest_unfunded_units:,.0f} units).
  Move the slider to explore the stored budget scenarios.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(key="clone-procurement-metrics"):
        metric_columns = st.columns(4)
        metric_specs = [
            (
                "Total units needed", f"{required_units:,.0f}",
                "Point forecast requirement after planning stock",
                "Planning data", "info", "default",
            ),
            (
                "Scenario budget", format_peso(selected_budget_value),
                f"{overview_budget:.0f}% of the full point budget",
                "Planning data", "info", "teal",
            ),
            (
                "Units funded", f"{allocated_units:,.0f}",
                f"Stored allocation at {overview_budget:.0f}% budget",
                "Planning data", "info", "default",
            ),
            (
                "Units not funded", f"{unfunded_units:,.0f}",
                "Review the allocation gap" if selected_has_gap else "All requirements covered",
                "Needs review" if selected_has_gap else "Covered",
                "attention" if selected_has_gap else "success",
                "danger" if selected_has_gap else "default",
            ),
        ]
        for column, spec in zip(metric_columns, metric_specs):
            label, value, description, badge, badge_tone, value_tone = spec
            with column:
                workspace_metric_card(
                    label, value, description, "forecast", badge,
                    badge_tone=badge_tone, value_tone=value_tone,
                )

    with clone_section("Budget scenario summary", "procurement-budget-summary"):
        with st.container(key="clone-procurement-budget-control"):
            budget_label, budget_value = st.columns([3, 1])
            with budget_label:
                st.markdown("**Budget level**")
            with budget_value:
                st.markdown(
                    f'<div class="dss-clone-budget-value">{overview_budget:.0f}%</div>',
                    unsafe_allow_html=True,
                )
            if point_budgets == [60.0, 70.0, 80.0, 90.0, 100.0]:
                st.slider(
                    "Budget level", min_value=60, max_value=100, step=10,
                    key=budget_key, format="%d%%", label_visibility="collapsed",
                )
            else:
                st.select_slider(
                    "Budget level", options=point_budgets, key=budget_key,
                    format_func=lambda value: f"{value:.0f}%",
                    label_visibility="collapsed",
                )
            st.markdown(
                '<div class="dss-clone-budget-endpoints">'
                f'<span>{lowest_budget:.0f}% · constrained</span>'
                f'<span>{full_budget:.0f}% · full budget</span></div>',
                unsafe_allow_html=True,
            )

        scenario_rows = []
        for candidate_budget in point_budgets:
            candidate_overview = (
                reference_budget_results[candidate_budget]["overview"].iloc[0]
            )
            scenario_rows.append({
                "Planning_Scenario": (
                    "Point forecast" if str(reference_policy).upper() == "POINT"
                    else str(reference_policy)
                ),
                "Budget_Percent": candidate_budget,
                "Scenario_Budget": float(candidate_overview["Scenario_Budget"]),
                "Required_Units": float(candidate_overview["Required_Units"]),
                "Allocated_Units": float(candidate_overview["Allocated_Units"]),
                "Unfunded_Requirement_Units": float(
                    candidate_overview["Unfunded_Requirement_Units"]
                ),
                "Evidence": "Stored optimizer result",
            })
        clone_table(
            pd.DataFrame(scenario_rows),
            "procurement-budget-scenarios",
            labels={
                "Planning_Scenario": "Planning scenario",
                "Budget_Percent": "Budget level",
                "Scenario_Budget": "Budget amount",
                "Required_Units": "Units needed",
                "Allocated_Units": "Units funded",
                "Unfunded_Requirement_Units": "Units not funded",
            },
            page_size=8,
        )
        st.caption(
            "Every budget level above is a stored optimizer result. "
            "Changing the selection does not interpolate allocations, rerun the "
            "optimizer, or authorize a purchase."
        )

    with clone_section(
        f"Medicine allocation detail · {overview_budget:.0f}% budget",
        "procurement-allocation-detail",
    ):
        allocation_search_col, allocation_count_col, unfunded_col = (
            st.columns([4.5, 1, 1.3], vertical_alignment="center")
        )
        with allocation_search_col:
            allocation_search = st.text_input(
                "Search medicine allocation",
                placeholder="Search brand, generic name, or medicine key...",
                key="procurement_allocation_search",
                label_visibility="collapsed",
            )
        with unfunded_col:
            unfunded_only = st.toggle(
                "Unfunded only", key="procurement_clone_unfunded_only",
            )

        allocation_preview = selected_result["allocations"].copy()
        if allocation_search.strip():
            lookup = allocation_search.strip()
            search_columns = [
                column for column in ("Brand_Name", "Generic_Name", "Medicine_Key")
                if column in allocation_preview.columns
            ]
            allocation_mask = pd.Series(False, index=allocation_preview.index)
            for search_column in search_columns:
                allocation_mask |= allocation_preview[search_column].fillna("").astype(str).str.contains(
                    lookup, case=False, regex=False,
                )
            allocation_preview = allocation_preview[allocation_mask]
        if unfunded_only:
            allocation_preview = allocation_preview[
                allocation_preview["Unfunded_Requirement_Units"] > 0
            ]

        with allocation_count_col:
            st.markdown(f"**{len(allocation_preview):,}** medicines")
        allocation_columns = [
            column for column in (
                "Brand_Name", "Generic_Name", "Unit", "ABC_AsOf",
                "Net_Requirement", "Allocated_Units",
                "Unfunded_Requirement_Units", "Item_Coverage_Ratio",
            )
            if column in allocation_preview.columns
        ]
        allocation_display = allocation_preview[allocation_columns].copy()
        clone_table(
            allocation_display,
            "procurement-allocations",
            labels={
                "Brand_Name": "Brand name",
                "Generic_Name": "Generic name",
                "ABC_AsOf": "ABC class",
                "Net_Requirement": "Recommended buy",
                "Allocated_Units": "Units funded",
                "Unfunded_Requirement_Units": "Not funded",
                "Item_Coverage_Ratio": "Funded coverage",
            },
            page_size=8,
        )
        st.caption(
            f"Showing {len(allocation_preview):,} of "
            f"{len(selected_result['allocations']):,} modeled medicines. "
            "Search and the unfunded filter affect the table only, not the "
            "portfolio totals or the optimizer results."
        )
        with st.expander("Show all technical fields", expanded=False):
            display_raw_table(allocation_preview)

    st.markdown(
        f"""
<div class="dss-clone-callout attention">
  <span class="dss-clone-callout-icon">{icon_svg("attention", "#b8741a", 22)}</span>
  <div><strong>Procurement readiness overview</strong>
  <p>Review medicines with partial or no funding under the selected scenario.
  Authorized staff must decide whether to revise requirements or budget.
  This tool provides planning evidence; it does not authorize purchase orders.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    section_title(
        "Detailed procurement workspace"
    )



    p1, p2, p3, p4 = st.columns(4)



    p1.metric(
        "Planning Medicines",
        f"{summary['Planning_Medicines']:,}",
    )


    p2.metric(
        "Optimizer Ready",
        f"{summary['Optimizer_Ready']:,}",
    )


    p3.metric(
        "Stock Review Required",
        f"{summary['Needs_Stock_Override']:,}",
    )


    p4.metric(
        "Cost Review Required",
        f"{summary['Needs_Cost_Override']:,}",
    )



    st.divider()



    tab1, tab2, tab3 = st.tabs(
        [
            "Requirement Planning",
            "Budget Optimizer",
            "Evidence",
        ]
    )



    with tab1:


        section_title(
            "Scenario-based procurement requirement"
        )



        scenario_df = pd.DataFrame(

            {

                "Scenario":
                [
                    "Point",
                    "P80",
                    "P90",
                ],

                "Requirement":
                [
                    summary["Point_Net_Requirement"],
                    summary["P80_Net_Requirement"],
                    summary["P90_Net_Requirement"],
                ],

                "Estimated Value":
                [
                    summary["Point_Value"],
                    summary["P80_Value"],
                    summary["P90_Value"],
                ],

            }

        )



        display_table(scenario_df)

        display_table(
            medicines,
            columns=[
                "Brand_Name",
                "Generic_Name",
                "Unit",
                "ABC_AsOf",
                "Behavior_Segment_AsOf",
                "Net_Requirement_Point",
                "Net_Requirement_P80",
                "Net_Requirement_P90",
                "Estimated_Procurement_Value_Point",
                "Estimated_Procurement_Value_P80",
                "Estimated_Procurement_Value_P90",
                "Planning_Input_Status",
            ],
        )

        with st.expander("Show all technical fields", expanded=False):
            display_raw_table(medicines)



    with tab2:


        status_message(
            "Validated optimizer outputs are displayed "
            "without rerunning the optimization model.",
            "info",
        )



        c1, c2 = st.columns(2)



        with c1:

            selected_policy = st.selectbox(
                "Requirement Policy",
                policies,
                format_func=lambda v: {
                    "POINT": "Standard (Point Forecast)",
                    "P80_FLOORED_CEIL": "Safer (P80 Buffer)",
                    "P90_FLOORED_CEIL": "Safest (P90 Buffer)",
                }.get(v, v),
                key="optimizer_policy_phase21",
            )



        with c2:

            selected_budget = st.selectbox(
                "Budget Level",
                budgets,
                format_func=lambda v: f"{int(v)}%",
                key="optimizer_budget_phase21",
            )



        result = get_procurement_optimizer_result(
            selected_policy,
            selected_budget,
        )



        display_table(
            result["overview"],
            columns=[
                "Requirement_Scenario",
                "Budget_Percent",
                "Scenario_Budget",
                "Required_Units",
                "Allocated_Units",
                "Unfunded_Requirement_Units",
                "Allocated_Value",
                "Budget_Utilization_Pct",
            ],
        )


        display_table(
            result["allocations"],
            columns=[
                "Brand_Name",
                "Generic_Name",
                "Unit",
                "ABC_AsOf",
                "Net_Requirement",
                "Allocated_Units",
                "Unfunded_Requirement_Units",
                "Allocated_Value",
                "Item_Coverage_Ratio",
            ],
        )

        with st.expander("Show all technical fields", expanded=False):
            display_raw_table(result["allocations"])



    with tab3:


        evidence_name = st.selectbox(
            "Technical Evidence Category",
            list(PROCUREMENT_EVIDENCE_TABLES.keys()),
            format_func=lambda value: PROCUREMENT_EVIDENCE_LABELS.get(
                value,
                value,
            ),
            key="procurement_evidence_phase21",
        )


        evidence = get_procurement_evidence_table(
            evidence_name
        )


        display_evidence_table(
            evidence,
            hidden_columns=PROCUREMENT_TECHNICAL_COLUMNS,
        )



    decision_guardrail(
        "Procurement interpretation",
        (
            "Procurement outputs represent modeled "
            "planning recommendations based on "
            "forecast assumptions, stock inputs, "
            "costs, and budget constraints."
        ),
    )

# ============================================================
# DATA QUALITY & EXPIRY REVIEW
# ============================================================

def render_expiry_data_quality() -> None:
    workspace_page_header(
        "Expiry & Data Quality Review",
        "Review historical evidence and resolve data gaps before purchasing.",
    )
    try:
        summary = get_expiry_dss_summary()
        explorer = get_expiry_medicine_explorer()
    except Exception as error:
        st.error("Expiry module unavailable.")
        st.exception(error)
        return

    flagged_expiry_count = int(
        explorer["Has_Authoritative_Expiry_History"].fillna(False).sum()
    )
    expiry_display_columns = [
        "Brand_Name",
        "Generic_Name",
        "Unit",
        "Expiry_History_Coverage_Status",
        "Expiry_History_Data_Quality_Status",
        "Latest_Expiration_Date_Represented",
        "Expiry_DSS_Interpretation",
    ]
    render_question_panel(
        "expiry",
        [
            {
                "key": "expiry",
                "icon": "expiry",
                "text": "Are we buying medicines that might expire?",
                "severity": (
                    "attention"
                    if flagged_expiry_count > 0
                    else "ok"
                ),
                "summary": (
                    f"{flagged_expiry_count:,} medicines have linked historical expiry "
                    "evidence. Review the records; history does not predict current expiry risk."
                ),
                "interpretation": (
                    f"{flagged_expiry_count:,} medicines have historical "
                    "expiry evidence on record. This does not predict future "
                    "expiry; it flags medicines with a documented history "
                    "that should be reviewed before procurement."
                ),
                "render": lambda: display_table(
                    explorer,
                    columns=expiry_display_columns,
                ),
            },
            {
                "key": "quality",
                "icon": "governance",
                "text": "Is there anything wrong with the underlying data?",
                "severity": "unavailable",
                "badge_label": "◷ Not available yet",
                "summary": (
                    "A complete automated quality assessment is not available. "
                    "Review the existing flags and identity gaps."
                ),
                "interpretation": (
                    "Data quality flagging beyond expiry evidence is not yet "
                    "available in this system."
                ),
                "render": lambda: None,
            },
        ],
    )


    with st.container(key="clone-expiry-kpis"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            workspace_metric_card(
                "Planning medicines", f"{summary['Planning_Medicines']:,}",
                f"{PLANNING_YEAR} planning cycle", "medicine", "ⓘ Planning data",
            )
        with c2:
            workspace_metric_card(
                "Linked history", f"{summary['With_Expiry_History']:,}",
                "Medicines with expiry evidence", "expiry", "ⓘ Planning data",
            )
        with c3:
            workspace_metric_card(
                "No linked history", f"{summary['Without_Expiry_History']:,}",
                "Evidence gaps to review", "forecast", "⚠ Coverage gap",
                badge_tone="attention",
            )
        with c4:
            workspace_metric_card(
                "Unresolved identities", f"{summary['Unresolved_Expiry_Identities']:,}",
                "Resolve medicine matching issues", "forecast", "⚠ Resolve",
                badge_tone="attention",
            )



    tab1, tab2 = st.tabs(
        [
            "Medicine Review",
            "Evidence Tables",
        ]
    )



    with tab1:


        search = st.text_input(
            "Search medicine",
            key="expiry_search_phase21",
        )


        filtered = explorer.copy()



        if search.strip():

            value = search.strip()


            mask = (

                filtered["Brand_Name"]
                .fillna("")
                .str.contains(
                    value,
                    case=False,
                    regex=False,
                )

                |

                filtered["Generic_Name"]
                .fillna("")
                .str.contains(
                    value,
                    case=False,
                    regex=False,
                )

            )


            filtered = filtered[mask]



        with clone_section("Historical expiry evidence", "expiry-medicine-review"):
            st.caption(f"{len(filtered):,} matching medicines")
            clone_table(
                filtered[expiry_display_columns], key="expiry-medicine-review-table",
                labels=COLUMN_LABELS,
            )

        with st.expander("Show all technical fields", expanded=False):
            display_raw_table(
                filtered.drop(
                    columns=[
                        "Brand_Name_1",
                        "Generic_Name_1",
                        "Unit_1",
                    ],
                    errors="ignore",
                ),
            )



    with tab2:


        evidence_name = st.selectbox(
            "Technical Evidence Category",
            list(EXPIRY_EVIDENCE_TABLES.keys()),
            format_func=lambda value: EXPIRY_EVIDENCE_LABELS.get(
                value,
                value,
            ),
            key="expiry_evidence_phase21",
        )



        evidence = get_expiry_evidence_table(
            evidence_name
        )



        display_evidence_table(
            evidence,
            expiry_view=True,
        )



    decision_guardrail(
        "Expiry interpretation",
        (
            "Historical expiry evidence supports "
            "data-quality review and governance. "
            "It does not measure current expiry exposure."
        ),
    )



# ============================================================
# TECHNICAL VALIDATION
# ============================================================

def render_retrospective_validation() -> None:


    page_header(
        "Retrospective Validation",
        "Retrospective Planning Validation",
        (
            "Evaluate historical planning policies "
            "using controlled retrospective evidence."
        ),
    )



    try:

        core = get_retrospective_core_tables()


    except Exception as error:

        st.error(
            "Validation outputs unavailable."
        )

        st.exception(error)

        return



    summary = core["summary"]



    section_title(
        "Validation overview"
    )



    c1, c2, c3 = st.columns(3)



    c1.metric(
        "Validation Cases",
        f"{len(summary):,}",
    )


    c2.metric(
        "Policy Variants",
        f"{summary.shape[1]:,}",
    )


    c3.metric(
        "Evidence Tables",
        f"{len(core):,}",
    )



    tab1, tab2 = st.tabs(
        [
            "Validation Results",
            "Technical Evidence",
        ]
    )



    with tab1:


        display_raw_table(summary)


        status_message(
            "Actual outcomes are introduced only "
            "during retrospective evaluation.",
            "info",
        )



    with tab2:


        evidence_name = st.selectbox(
            "Technical Evidence Category",
            list(RETROSPECTIVE_EVIDENCE_TABLES.keys()),
            format_func=lambda value: RETROSPECTIVE_EVIDENCE_LABELS.get(
                value,
                value,
            ),
            key="validation_evidence_phase21",
        )


        evidence = get_retrospective_evidence_table(
            evidence_name
        )


        display_raw_table(evidence)



    decision_guardrail(
        "Validation interpretation",
        (
            "Retrospective outputs evaluate framework "
            "behavior and do not reconstruct clinical outcomes."
        ),
    )



# ============================================================
# METHODOLOGY & GOVERNANCE
# ============================================================

def render_governance_methodology() -> None:


    page_header(
        None,
        "Methodology & Governance",
        "Understand how the system works and who is responsible for each decision.",
    )



    overview = get_governance_overview()

    historical_period = (
        f"{overview['Historical_Start_Year']}&ndash;"
        f"{overview['Historical_End_Year']}"
    )

    st.markdown(
        """
<section class="dss-governance-intro">
  <div class="dss-governance-intro-title">ℹ&nbsp;&nbsp; About this system</div>
  <div class="dss-governance-intro-body">
    CDH Inventory DSS is a decision-support tool: it provides recommendations
    and evidence, not automatic decisions. Procurement and planning decisions
    remain with authorized pharmacy staff following the governed workflow below.
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<section class="dss-governance-panel">
  <div class="dss-governance-panel-title">Methodology overview</div>
  <details class="dss-method-item" open><summary class="dss-method-row first"><span class="dss-method-number">1</span><span>Demand forecasting approach</span><span class="dss-method-action" aria-hidden="true"></span></summary><div class="dss-method-body">
    Historical medicine issuance from {historical_period} establishes the
    planning baseline. Governed forecasting produces Point, P80, and P90
    planning scenarios after medicines are segmented by observed demand behavior.
  </div></details>
  <details class="dss-method-item"><summary class="dss-method-row"><span class="dss-method-number">2</span><span>ABC economic classification</span><span class="dss-method-action" aria-hidden="true"></span></summary><div class="dss-method-body">Medicines are grouped by annual consumption value. ABC is an economic planning priority and does not measure clinical importance. Missing cost information remains unclassified.</div></details>
  <details class="dss-method-item"><summary class="dss-method-row"><span class="dss-method-number">3</span><span>Demand pattern classification</span><span class="dss-method-action" aria-hidden="true"></span></summary><div class="dss-method-body">Observed issuance history is segmented as persistent stable, persistent variable, sparse, or non-moving. Demand behavior and ABC economic class are independent dimensions.</div></details>
  <details class="dss-method-item"><summary class="dss-method-row"><span class="dss-method-number">4</span><span>Expiry evidence methodology</span><span class="dss-method-action" aria-hidden="true"></span></summary><div class="dss-method-body">Linked expiry records are historical evidence for review and governance. They do not show current expiring stock or predict current expiry risk.</div></details>
  <details class="dss-method-item"><summary class="dss-method-row"><span class="dss-method-number">5</span><span>Planning-input quality review</span><span class="dss-method-action" aria-hidden="true"></span></summary><div class="dss-method-body">Stock, unit-cost, identity, and forecast-input checks identify medicines requiring staff review before the procurement plan is finalized. The DSS provides evidence; authorized staff make the decisions.</div></details>
</section>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<section class="dss-governance-panel">
  <div class="dss-governance-panel-title">Decision governance &mdash; who does what</div>
  <div class="dss-governance-table-wrap">
    <table class="dss-governance-table">
      <thead><tr><th>Role</th><th>Responsibility</th></tr></thead>
      <tbody>
        <tr><td>Pharmacy Director</td><td>Approves the final procurement plan and budget-allocation decisions.</td></tr>
        <tr><td>Chief Pharmacist</td><td>Reviews planning assumptions, ABC classification, and demand patterns.</td></tr>
        <tr><td>Procurement Officer</td><td>Executes approved procurement actions and tracks delivery.</td></tr>
        <tr><td>Data Analyst</td><td>Resolves data-quality issues, maintains historical evidence, and runs governed DSS processes.</td></tr>
        <tr><td>Finance Officer</td><td>Confirms budget availability and reviews the selected planning scenario.</td></tr>
      </tbody>
    </table>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<section class="dss-governance-panel">
  <div class="dss-governance-panel-title">Planning cycle timeline</div>
  <div class="dss-timeline">
    <div class="dss-timeline-item"><div class="dss-timeline-marker"><span class="dss-timeline-number">1</span></div><div class="dss-timeline-content"><div class="dss-timeline-month">November</div><div class="dss-timeline-title">Data extraction and quality review</div><div class="dss-timeline-owner">Owner: Data Analyst</div></div></div>
    <div class="dss-timeline-item"><div class="dss-timeline-marker"><span class="dss-timeline-number">2</span></div><div class="dss-timeline-content"><div class="dss-timeline-month">December</div><div class="dss-timeline-title">ABC classification update and demand pattern review</div><div class="dss-timeline-owner">Owner: Chief Pharmacist</div></div></div>
    <div class="dss-timeline-item"><div class="dss-timeline-marker"><span class="dss-timeline-number">3</span></div><div class="dss-timeline-content"><div class="dss-timeline-month">January</div><div class="dss-timeline-title">Forecast generation and review</div><div class="dss-timeline-owner">Owner: Pharmacy Director + Data Analyst</div></div></div>
    <div class="dss-timeline-item"><div class="dss-timeline-marker"><span class="dss-timeline-number">4</span></div><div class="dss-timeline-content"><div class="dss-timeline-month">February</div><div class="dss-timeline-title">Budget scenario analysis and procurement plan finalization</div><div class="dss-timeline-owner">Owner: Finance Officer + Procurement Officer</div></div></div>
    <div class="dss-timeline-item"><div class="dss-timeline-marker"><span class="dss-timeline-number">5</span></div><div class="dss-timeline-content"><div class="dss-timeline-month">March</div><div class="dss-timeline-title">Approved procurement plan submitted to suppliers</div><div class="dss-timeline-owner">Owner: Procurement Officer</div></div></div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    source_col, validation_col = st.columns(2)

    with source_col:
        st.markdown(
            f"""
<section class="dss-evidence-card source">
  <div class="dss-evidence-card-title">ℹ&nbsp;&nbsp; Data sources</div>
  <div class="dss-evidence-card-body">
    The published analytical snapshot uses governed historical inventory and
    planning evidence from {historical_period}. Source provenance and data
    limitations are retained in the supporting evidence tables.
  </div>
</section>
""",
            unsafe_allow_html=True,
        )

    with validation_col:
        st.markdown(
            """
<section class="dss-evidence-card validation">
  <div class="dss-evidence-card-title">✓&nbsp;&nbsp; Model validation status</div>
  <div class="dss-evidence-card-body">
    Forecast model selection, quantile validation, procurement optimization,
    and retrospective checks are documented in governance evidence records.
    These results support planning; they do not replace formal approval.
  </div>
</section>
""",
            unsafe_allow_html=True,
        )

    return



    c1, c2, c3, c4 = st.columns(4)



    c1.metric(
        "Historical Period",
        (
            f"{overview['Historical_Start_Year']}"
            "-"
            f"{overview['Historical_End_Year']}"
        ),
    )


    c2.metric(
        "Planning Year",
        overview["Planning_Target_Year"],
    )


    c3.metric(
        "Medicines",
        f"{overview['Medicine_Count']:,}",
    )


    c4.metric(
        "Inventory Rows",
        f"{overview['Annual_Inventory_Rows']:,}",
    )



    st.divider()



    st.markdown(
        f"""
## Analytical workflow

### 1. Historical foundation
Medicine-year inventory records from
**{HISTORICAL_PERIOD_LABEL}**
establish the observed consumption baseline.

### 2. Demand segmentation
Medicines are grouped using ABC classification
and demand behavior analysis.

### 3. Forecasting
Forecast scenarios are generated from historical evidence.

### 4. Procurement planning
Forecast outputs are transformed into requirements.

### 5. Optimization
Budget allocation is evaluated using constraints.

### 6. Validation
Historical simulations evaluate policy behavior.

### 7. Governance
Evidence tables preserve provenance and limitations.
"""
    )



    section_title(
        "Analytical Decision Workflow"
    )


    workflow_col1, workflow_col2 = st.columns(2)


    with workflow_col1:

        workflow_step(
            1,
            "Historical Inventory",
            "Uses completed annual medicine records.",
        )

        workflow_step(
            2,
            "Demand Profiling",
            "Identifies usage behavior and economic class.",
        )

        workflow_step(
            3,
            "Forecasting",
            "Generates planning scenarios.",
        )


    with workflow_col2:

        workflow_step(
            4,
            "Procurement Planning",
            "Transforms forecasts into requirements.",
        )

        workflow_step(
            5,
            "Optimization",
            "Evaluates budget allocation.",
        )

        workflow_step(
            6,
            "Governance Review",
            "Preserves evidence and limitations.",
        )


    decision_guardrail(
        "Governance principle",
        (
            "The DSS separates observed evidence "
            "from modeled inference and preserves "
            "analytical limitations."
        ),
    )



# ============================================================
# ROUTING
# ============================================================


if page == "Planning Overview":

    render_executive_dashboard()



elif page == "Medicine Prioritization":

    render_inventory_demand_profile()



elif page == "Forecast Planning":

    render_forecasting_uncertainty()



elif page == "Procurement Planning":

    render_procurement_decision_support()



elif page == "Expiry & Data Quality Review":

    render_expiry_data_quality()



elif page == "Retrospective Validation":

    render_retrospective_validation()



elif page == "Methodology & Governance":

    render_governance_methodology()



clone_footer()
