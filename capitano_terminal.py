"""
GEXRADAR // QUANT TERMINAL  — Premium Redesign
Run: streamlit run app.py

─────────────────────────────────────────────────────────────────────────────
GEX FORMULA (Perfiliev / SpotGamma / Barchart industry standard):
  GEX = Gamma × OI × ContractSize(100) × Spot² × 0.01 / 1e9

  This gives dealer dollar-exposure per 1% spot move, in billions.
  Calls: +GEX  (dealers assumed long calls → long gamma → stabilising)
  Puts:  -GEX  (dealers assumed short puts → short gamma → destabilising)

FIXES APPLIED vs prior version:
  1. Strike range aligned to ±8% in BOTH _process_chain AND bar_layout.
     (Old ±12% in calc + ±8% chart = phantom OOB contributions to cum-sum.)
  2. OI minimum raised 10 → 100. At OI=10 near-expiry ATM gamma is enormous;
     tiny positions made prominent bars. 100 is consistent with SpotGamma.
  3. IV gating: require bid>0 AND ask>0 AND mid>0.05 before solving IV.
     Zero-bid OTM options previously got a full smile fallback IV applied,
     producing phantom GEX bars with no real market behind them.
  4. Fallback IV now ONLY applied when mid>0.05 (real market exists).
     No valid IV + no real market → skip the strike entirely.
  5. IV cap tightened 2.5→1.5. 250% IV on equity options is unrealistic
     and could still inflate gamma on edge-case strikes.
  6. Heatmap fetch_options_data_heatmap: added 90 DTE cap (was missing).
  7. ES/NQ equiv conversion: now uses DYNAMIC live ratio (ES_price/SPY_price,
     NQ_price/QQQ_price) fetched from Yahoo Finance, NOT a hardcoded multiplier.
     Ratio floats daily with dividends, carry, and index drift. Falls back
     to last known ratio if fetch fails.
  8. Auto-refresh: page reruns every 60 seconds to keep GEX levels current.
  9. Cache TTL reduced from 900s to 60s to prevent stale data on cloud deployments.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import time
import datetime
import warnings
import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.043  # ~Fed funds rate as of early 2025; update as rates change
DIV_YIELD = {
    "SPY": 0.013, "QQQ": 0.006, "IWM": 0.012,
    "GLD": 0.0,   "SLV": 0.0,   "TLT": 0.04,
    "XLF": 0.018, "XLE": 0.035, "IBIT": 0.0,
    "AAPL": 0.005,"NVDA": 0.001,"TSLA": 0.0,
    "AMZN": 0.0,  "MSFT": 0.007,"META": 0.004,
    "GOOGL": 0.0, "SPX": 0.013, "NDX": 0.006,
    "RUT": 0.012,
}

# Auto-refresh interval in seconds
AUTO_REFRESH_SECONDS = 60

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CAPITANO TERMINAL", layout="wide", initial_sidebar_state="expanded")

if "current_page" not in st.session_state:
    st.session_state.current_page = "DASHBOARD"
if "radar_mode" not in st.session_state:
    st.session_state.radar_mode = "GEX"
if "asset_choice" not in st.session_state:
    st.session_state.asset_choice = "SPY"
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

st.session_state.current_page = "DASHBOARD"
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Default"


# ─────────────────────────────────────────────────────────────────────────────
# THEME SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    # ── Original dark navy/teal ─────────────────────────────────────────────
    "Default": dict(
        bg="#03070D", bg1="#070E17", bg2="#0B1420", bg3="#0F1C2E",
        line="#111E2E", line2="#192D44", line_bright="#1F3A58",
        t1="#C8DCEF",  t2="#7FA8C4",  t3="#4E7A9C",  t4="#2A4260",
        green="#00E5A0", red="#FF3860", amber="#F5A623", blue="#4A9FFF", violet="#9B7FFF",
        green_glow="rgba(0,229,160,0.10)", red_glow="rgba(255,56,96,0.10)",
        nav_bg="rgba(3,7,13,0.97)", nav_border="rgba(255,255,255,0.05)",
        nav_t1="#C8DCEF", nav_t2="#3D6680", nav_clock="#5A8AA8", nav_dot="#00E5A0",
        chart_bg="#03070D", chart_line="#111E2E", chart_line2="#192D44",
        chart_t2="#7FA8C4", chart_t3="#4E7A9C",
        chart_hover="#0F1C2E", chart_hover_border="#192D44", chart_hover_text="#C8DCEF",
        bar_pos="#00E5A0", bar_neg="#FF3860", spot_line="#C8DCEF", gamma_line="#4E7A9C",
        surface_colorscale=[
            [0.00,"#FF0040"],[0.15,"#CC0030"],[0.30,"#7A0020"],
            [0.45,"#1A0308"],[0.50,"#03070D"],[0.55,"#001A0A"],
            [0.70,"#006633"],[0.85,"#00CC77"],[1.00,"#00FF99"],
        ],
        heat_colorscale=[
            [0.00,"#FF0040"],[0.20,"#CC0030"],[0.38,"#550015"],
            [0.48,"#150005"],[0.50,"#03070D"],[0.52,"#001505"],
            [0.62,"#005530"],[0.80,"#00CC70"],[1.00,"#00FF99"],
        ],
    ),
    # ── Pure black with green/red signals ──────────────────────────────────
    "Obsidian": dict(
        bg="#000000", bg1="#080808", bg2="#101010", bg3="#181818",
        line="rgba(255,255,255,0.06)", line2="rgba(255,255,255,0.11)", line_bright="rgba(255,255,255,0.22)",
        t1="#F0F0F0",  t2="#888888",  t3="#4A4A4A",  t4="#2A2A2A",
        green="#00FF88", red="#FF3355", amber="#FFB800", blue="#5599FF", violet="#AA88FF",
        green_glow="rgba(0,255,136,0.08)", red_glow="rgba(255,51,85,0.08)",
        nav_bg="#000000", nav_border="rgba(255,255,255,0.07)",
        nav_t1="#F0F0F0", nav_t2="#4A4A4A", nav_clock="#777777", nav_dot="#00FF88",
        chart_bg="#000000", chart_line="rgba(255,255,255,0.06)", chart_line2="rgba(255,255,255,0.11)",
        chart_t2="#888888", chart_t3="#444444",
        chart_hover="#111111", chart_hover_border="rgba(255,255,255,0.12)", chart_hover_text="#F0F0F0",
        bar_pos="#00FF88", bar_neg="#FF3355", spot_line="#F0F0F0", gamma_line="#333333",
        surface_colorscale=[
            [0.00,"#FF3355"],[0.20,"#CC1133"],[0.40,"#440011"],
            [0.50,"#000000"],[0.60,"#004422"],[0.80,"#00CC66"],[1.00,"#00FF88"],
        ],
        heat_colorscale=[
            [0.00,"#FF3355"],[0.25,"#881122"],[0.45,"#220008"],
            [0.50,"#000000"],[0.55,"#002211"],[0.75,"#008844"],[1.00,"#00FF88"],
        ],
    ),
    # ── Clean white — white/gray backgrounds, real signal colors ─────────────
    "Light": dict(
        bg="#FFFFFF", bg1="#F5F5F5", bg2="#EBEBEB", bg3="#E0E0E0",
        line="rgba(0,0,0,0.08)", line2="rgba(0,0,0,0.15)", line_bright="rgba(0,0,0,0.32)",
        t1="#0A0A0A",  t2="#444444",  t3="#888888",  t4="#BBBBBB",
        green="#00A86B", red="#D93025", amber="#B8860B", blue="#1A56CC", violet="#7B42A8",
        green_glow="rgba(0,168,107,0.12)", red_glow="rgba(217,48,37,0.12)",
        nav_bg="#FFFFFF", nav_border="rgba(0,0,0,0.09)",
        nav_t1="#0A0A0A", nav_t2="#888888", nav_clock="#555555", nav_dot="#00A86B",
        chart_bg="#FFFFFF", chart_line="rgba(0,0,0,0.08)", chart_line2="rgba(0,0,0,0.15)",
        chart_t2="#555555", chart_t3="#999999",
        chart_hover="#F0F0F0", chart_hover_border="rgba(0,0,0,0.18)", chart_hover_text="#0A0A0A",
        bar_pos="#00A86B", bar_neg="#D93025", spot_line="#0A0A0A", gamma_line="#AAAAAA",
        surface_colorscale=[
            [0.00,"#D93025"],[0.25,"#E8776F"],[0.48,"#F5D5D3"],
            [0.50,"#FFFFFF"],[0.52,"#D4EDE3"],[0.75,"#66C2A5"],[1.00,"#00A86B"],
        ],
        heat_colorscale=[
            [0.00,"#D93025"],[0.25,"#E8776F"],[0.47,"#F5E5E4"],
            [0.50,"#FFFFFF"],[0.53,"#E0F0E8"],[0.75,"#55BB8A"],[1.00,"#00A86B"],
        ],
    ),
    # ── Warm amber phosphor terminal ────────────────────────────────────────
    "Amber": dict(
        bg="#060400", bg1="#0E0A00", bg2="#161000", bg3="#1E1600",
        line="rgba(255,180,0,0.09)", line2="rgba(255,180,0,0.17)", line_bright="rgba(255,180,0,0.35)",
        t1="#FFE066",  t2="#CC9900",  t3="#7A5A00",  t4="#3D2D00",
        green="#FFD700", red="#FF6600", amber="#FFB800", blue="#FFCC44", violet="#FF9933",
        green_glow="rgba(255,215,0,0.08)", red_glow="rgba(255,102,0,0.08)",
        nav_bg="#060400", nav_border="rgba(255,180,0,0.10)",
        nav_t1="#FFE066", nav_t2="#7A5A00", nav_clock="#CC9900", nav_dot="#FFB800",
        chart_bg="#060400", chart_line="rgba(255,180,0,0.09)", chart_line2="rgba(255,180,0,0.17)",
        chart_t2="#CC9900", chart_t3="#664400",
        chart_hover="#161000", chart_hover_border="rgba(255,180,0,0.2)", chart_hover_text="#FFE066",
        bar_pos="#FFD700", bar_neg="#FF6600", spot_line="#FFE066", gamma_line="#664400",
        surface_colorscale=[
            [0.00,"#FF6600"],[0.30,"#AA3300"],[0.48,"#221100"],
            [0.50,"#060400"],[0.52,"#221400"],[0.70,"#AA7700"],[1.00,"#FFD700"],
        ],
        heat_colorscale=[
            [0.00,"#FF6600"],[0.25,"#882200"],[0.47,"#1A0A00"],
            [0.50,"#060400"],[0.53,"#1A0E00"],[0.75,"#997700"],[1.00,"#FFD700"],
        ],
    ),
    # ── Deep arctic blue ────────────────────────────────────────────────────
    "Glacier": dict(
        bg="#030710", bg1="#060D1C", bg2="#0A1428", bg3="#0E1B34",
        line="rgba(100,160,230,0.09)", line2="rgba(100,160,230,0.16)", line_bright="rgba(100,160,230,0.32)",
        t1="#D8ECFF",  t2="#6A9EC8",  t3="#3A6090",  t4="#1A3660",
        green="#44CCFF", red="#FF4466", amber="#66AAFF", blue="#44CCFF", violet="#8866FF",
        green_glow="rgba(68,204,255,0.09)", red_glow="rgba(255,68,102,0.09)",
        nav_bg="#030710", nav_border="rgba(100,160,230,0.09)",
        nav_t1="#D8ECFF", nav_t2="#3A6090", nav_clock="#6A9EC8", nav_dot="#44CCFF",
        chart_bg="#030710", chart_line="rgba(100,160,230,0.09)", chart_line2="rgba(100,160,230,0.16)",
        chart_t2="#6A9EC8", chart_t3="#3A6090",
        chart_hover="#0A1428", chart_hover_border="rgba(100,160,230,0.2)", chart_hover_text="#D8ECFF",
        bar_pos="#44CCFF", bar_neg="#FF4466", spot_line="#D8ECFF", gamma_line="#3A6090",
        surface_colorscale=[
            [0.00,"#FF4466"],[0.20,"#CC1133"],[0.40,"#330011"],
            [0.50,"#030710"],[0.60,"#001A44"],[0.80,"#0088CC"],[1.00,"#44CCFF"],
        ],
        heat_colorscale=[
            [0.00,"#FF4466"],[0.25,"#881122"],[0.45,"#150018"],
            [0.50,"#030710"],[0.55,"#001022"],[0.75,"#005599"],[1.00,"#44CCFF"],
        ],
    ),
}
def get_theme():
    name = st.session_state.get("ui_theme", "Default")
    return THEMES.get(name, THEMES["Default"])


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
# ── Dynamic theme CSS injection ────────────────────────────────────────────
_T = get_theme()
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Barlow:wght@300;400;500;600&display=swap');
:root {{
    --bg:            {_T['bg']};
    --base:          {_T['bg1']};
    --surface:       {_T['bg2']};
    --surface-2:     {_T['bg2']};
    --surface-3:     {_T['bg3']};
    --void:          {_T['bg']};
    --line:          {_T['line']};
    --line-2:        {_T['line2']};
    --line-bright:   {_T['line_bright']};
    --text-1:        {_T['t1']};
    --text-2:        {_T['t2']};
    --text-3:        {_T['t3']};
    --text-4:        {_T['t4']};
    --green:         {_T['green']};
    --green-dim:     {_T['green']};
    --green-glow:    {_T['green_glow']};
    --red:           {_T['red']};
    --red-dim:       {_T['red']};
    --red-glow:      {_T['red_glow']};
    --amber:         {_T['amber']};
    --blue:          {_T['blue']};
    --violet:        {_T['violet']};
    --mono:          'JetBrains Mono', monospace;
    --display:       'Barlow Condensed', sans-serif;
    --body:          'Barlow', sans-serif;
    --radius:        2px;
    --radius-lg:     3px;
    --radius-xl:     4px;
    --transition:    all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  /* aliases for legacy class refs */
  --bg-1: {_T['bg1']};
  --bg-2: {_T['bg2']};
  --bg-3: {_T['bg3']};
}}
</style>
""", unsafe_allow_html=True)

# ── Update module-level Plotly vars ────────────────────────────────────────
_TC  = get_theme()
BG   = _TC["chart_bg"]
LINE = _TC["chart_line"]
LINE2= _TC["chart_line2"]
TEXT2= _TC["chart_t2"]
TEXT3= _TC["chart_t3"]


st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }
html, body, [class*="css"] {
    font-family: var(--body);
    background-color: var(--bg);
    color: var(--text-1);
    -webkit-font-smoothing: antialiased;
}

section[data-testid="stSidebar"] {
    background: var(--base) !important;
    border-right: 1px solid var(--line) !important;
    min-width: 240px !important;
    max-width: 240px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Show sidebar navigation"],
[data-testid="stSidebarNavCollapseButton"] {
    display: none !important;
    pointer-events: none !important;
}

.exp-dial-wrap { padding: 10px 0 6px 0; }
.exp-dial-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }
.exp-dial-label { font-family:var(--body); font-size:9px; color:var(--text-3); letter-spacing:1.5px; text-transform:uppercase; }
.exp-dial-value { font-family:var(--mono); font-size:18px; font-weight:500; color:var(--text-1); letter-spacing:-1px; line-height:1; }
.exp-pip-row { display:flex; gap:4px; align-items:flex-end; height:28px; }
.exp-pip { flex:1; border-radius:1px; transition:var(--transition); cursor:pointer; }
.exp-pip.active { background:var(--text-1); }
.exp-pip.inactive { background:var(--bg-3); border:1px solid var(--line); }
.exp-pip.inactive:hover { background:var(--bg-2); }
.dual-btn-wrap button {
    font-family: var(--mono) !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    letter-spacing: .8px !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--line2) !important;
    color: var(--text-3) !important;
    border-radius: 3px !important;
    padding: 6px 12px !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important;
    margin-bottom: 12px !important;
}
.dual-btn-wrap button:hover {
    border-color: var(--text-3) !important;
    color: var(--text-1) !important;
    background: var(--bg-2) !important;
}
.dual-btn-wrap button:active {
    transform: scale(.97) !important;
    transition: transform .04s ease !important;
}
.dual-btn-wrap button:focus,
.dual-btn-wrap button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}

.sb-section {
    font-family: var(--body);
    font-size: 8px;
    font-weight: 600;
    color: var(--text-4);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 20px 0 7px 0;
    margin: 0;
    border-bottom: 1px solid var(--line);
}
.sb-section:first-child { padding-top: 14px; }

.m-tile {
    padding: 7px 0;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    transition: var(--transition);
}
.m-tile:last-child { border-bottom: none; }
.m-label {
    font-family: var(--body);
    font-size: 9px;
    color: var(--text-3);
    letter-spacing: 0.4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 1;
    min-width: 0;
}
.m-value {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: -0.3px;
    white-space: nowrap;
    flex-shrink: 0;
}
.sb-group { padding: 4px 0 10px 0; }

.regime {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}
.regime::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    height: 1px;
    width: 100%;
    background: var(--bar-grad);
    opacity: 0.5;
}
.regime-meta { font-family: var(--mono); font-size: 9px; color: var(--text-3); letter-spacing: 1.2px; margin-bottom: 6px; text-transform: uppercase; }
.regime-state { font-family: var(--display); font-size: 22px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.regime-right { text-align: right; }
.regime-bias-label { font-family: var(--body); font-size: 8.5px; color: var(--text-3); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 5px; }
.regime-bias-value { font-family: var(--display); font-size: 15px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }

.mode-btn-row { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }
.mode-btn {
    font-family: var(--mono); font-size: 9px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
    padding: 7px 14px; border-radius: var(--radius); border: 1px solid var(--line);
    background: transparent; color: var(--text-3); cursor: pointer; transition: var(--transition);
    white-space: nowrap;
}
.mode-btn:hover { border-color: var(--line-bright); color: var(--text-1); background: var(--bg-2); }
.mode-btn.active { border-color: var(--text-1); color: var(--text-1); background: var(--bg-2); }

div.stButton > button {
    background: transparent !important;
    border: 1px solid var(--line2) !important;
    color: var(--text-3) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: .8px !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    height: 34px !important;
    width: 100% !important;
    cursor: pointer !important;
    user-select: none !important;
    white-space: nowrap !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important;
}
div.stButton > button:hover {
    background: var(--bg-2) !important;
    border-color: var(--text-3) !important;
    color: var(--text-1) !important;
}
div.stButton > button:active {
    transform: scale(.97) !important;
    transition: transform .04s ease !important;
}
div.stButton > button:focus,
div.stButton > button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
div.stButton > button[kind="primary"] {
    background: var(--bg-2) !important;
    border-color: var(--line-bright) !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover {
    border-color: var(--text-1) !important;
}
div.stButton > button[kind="primary"]:active {
    transform: scale(.97) !important;
}
/* kill the Streamlit spinner/loading overlay that flashes on button click */
div.stButton > button > div[data-testid="stSpinner"],
div.stButton > button svg,
div.stButton > button .stMarkdown { pointer-events: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

.kl-panel { background: var(--bg-1); border: 1px solid var(--line); border-radius: var(--radius-lg); overflow: hidden; }
.kl-header { padding: 11px 16px; border-bottom: 1px solid var(--line); font-family: var(--mono); font-size: 8px; font-weight: 600; color: var(--text-3); letter-spacing: 2.5px; text-transform: uppercase; background: var(--bg); display: flex; align-items: center; gap: 8px; }
.kl-header::before { content: ''; width: 14px; height: 1px; background: var(--text-1); flex-shrink: 0; }
.kl-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line); transition: var(--transition); cursor: default; }
.kl-row:hover { background: var(--bg-2); }
.kl-row:last-child { border-bottom: none; }
.kl-name { font-family: var(--body); font-size: 9.5px; color: var(--text-2); letter-spacing: 0.3px; }
.kl-val { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: -0.3px; }

.sec-head {
    font-family: var(--mono); font-size: 8px; font-weight: 600; color: var(--text-3);
    letter-spacing: 3px; text-transform: uppercase; padding: 22px 0 11px 0;
    border-bottom: 1px solid var(--line); margin-bottom: 16px; position: relative;
    display: flex; align-items: center; gap: 10px;
}
.sec-head::before { content: ''; width: 16px; height: 1px; background: var(--text-1); flex-shrink: 0; }

.sub-head {
    font-family: var(--mono); font-size: 8px; font-weight: 600; color: var(--text-3);
    letter-spacing: 2.5px; text-transform: uppercase; padding: 12px 0 9px 0;
    border-bottom: 1px solid var(--line); margin-bottom: 12px;
}

.stDataFrame { border: none !important; }
.stDataFrame thead tr th { font-family: var(--mono) !important; font-size: 8.5px !important; font-weight: 600 !important; color: var(--text-3) !important; letter-spacing: 1.8px !important; text-transform: uppercase !important; background: var(--bg) !important; border-bottom: 1px solid var(--line) !important; padding: 9px 11px !important; }
.stDataFrame tbody tr td { font-family: var(--mono) !important; font-size: 10.5px !important; padding: 8px 11px !important; border-bottom: 1px solid var(--line) !important; background: var(--base) !important; color: var(--text-1) !important; }
.stDataFrame tbody tr:hover td { background: var(--bg-2) !important; }

div[data-baseweb="select"] > div { background: var(--base) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important; font-family: var(--mono) !important; font-size: 11px !important; transition: var(--transition) !important; }
div[data-baseweb="select"] > div:hover { border-color: var(--line-bright) !important; }
div[data-baseweb="select"] span { color: var(--text-2) !important; font-family: var(--mono) !important; font-size: 11px !important; }
div[data-testid="stRadio"] label { font-family: var(--mono) !important; font-size: 11px !important; color: var(--text-2) !important; }

div[data-testid="stDownloadButton"] button {
    background: transparent !important; border: 1px solid var(--line2) !important;
    color: var(--text-3) !important; font-family: var(--mono) !important;
    font-size: 10px !important; font-weight: 600 !important; letter-spacing: .8px !important;
    text-transform: uppercase !important; border-radius: 3px !important;
    cursor: pointer !important; user-select: none !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important; }
div[data-testid="stDownloadButton"] button:hover { border-color: var(--text-3) !important; color: var(--text-1) !important; background: var(--bg-2) !important; }
div[data-testid="stDownloadButton"] button:active { transform: scale(.97) !important; transition: transform .04s ease !important; }
div[data-testid="stDownloadButton"] button:focus, div[data-testid="stDownloadButton"] button:focus-visible { outline: none !important; box-shadow: none !important; }

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 1px; }
::-webkit-scrollbar-thumb:hover { background: var(--line-bright); }

div[data-testid="stSpinner"] { color: var(--text-3) !important; }
div[data-testid="stSelectbox"] label { display: none !important; }

section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button {
    height: 30px !important; font-size: 11px !important; font-family: var(--mono) !important;
    font-weight: 600 !important; letter-spacing: 1.5px !important; border-radius: 20px !important;
    padding: 0 !important; user-select: none !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button:active { transform: scale(.96) !important; transition: transform .04s ease !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button:focus,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button:focus-visible { outline: none !important; box-shadow: none !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"] { background: var(--bg-3) !important; border-color: var(--line-bright) !important; color: var(--text-1) !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button[kind="secondary"] { background: transparent !important; border-color: var(--line) !important; color: var(--text-3) !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button[kind="secondary"]:hover { border-color: var(--line-bright) !important; color: var(--text-2) !important; }

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] { margin-top: 8px; }
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label { display: none !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div { border-radius: 20px !important; padding: 0 12px !important; min-height: 32px !important; height: 32px !important; font-family: var(--mono) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1.5px !important; background: var(--base) !important; border-color: var(--line-bright) !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] [data-testid="stWidgetLabel"] { outline: none !important; box-shadow: none !important; caret-color: transparent !important; animation: none !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] span { color: var(--text-1) !important; font-family: var(--mono) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1.5px !important; }
div[data-baseweb="popover"] ul[role="listbox"] li { font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: 1px !important; color: var(--text-2) !important; background: var(--base) !important; }
div[data-baseweb="popover"] ul[role="listbox"] li:hover { background: var(--bg-2) !important; color: var(--text-1) !important; }
div[data-baseweb="popover"] ul[role="listbox"] li[data-value^="──"] { font-family: var(--mono) !important; font-size: 9px !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--text-3) !important; background: #050505 !important; padding-top: 10px !important; padding-bottom: 4px !important; pointer-events: none !important; cursor: default !important; border-top: 1px solid var(--line) !important; }

.kl-row-full { display: flex; align-items: center; justify-content: space-between; padding: 9px 16px; border-bottom: 1px solid var(--line); transition: var(--transition); }
.kl-row-full:hover { background: var(--bg-2); }
.kl-row-full:last-child { border-bottom: none; }
.kl-row-full-left { display: flex; align-items: center; gap: 10px; }
.kl-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.kl-row-full .kl-name { font-family: var(--body); font-size: 10px; font-weight: 500; letter-spacing: 0.3px; }
.kl-price-val { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: -0.3px; }

.dual-toggle-row { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; }
.dual-pill { display: inline-flex; align-items: center; gap: 8px; background: transparent; border: 1px solid var(--line); border-radius: 24px; padding: 6px 14px 6px 10px; font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: 1.2px; color: var(--text-3); cursor: pointer; transition: var(--transition); text-transform: uppercase; user-select: none; }
.dual-pill:hover { border-color: var(--line-bright); color: var(--text-2); }
.dual-pill.active { border-color: var(--text-1); color: var(--text-1); background: var(--bg-2); }
.dual-pip { width: 6px; height: 6px; border-radius: 50%; background: var(--text-3); transition: var(--transition); }
.dual-pill.active .dual-pip { background: var(--text-1); }
.dual-pip-pair { display: flex; gap: 3px; }

/* ── Theme select pill — slightly larger & distinct from asset picker ── */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"]:first-of-type div[data-baseweb="select"] > div {
    border-radius: 3px !important;
    background: var(--bg) !important;
    border: 1px solid var(--line-bright) !important;
    letter-spacing: 2px !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"]:first-of-type div[data-baseweb="select"] span {
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-1) !important;
}

/* ── Pip bar glows with theme accent ──────────────────────────────── */
.exp-pip.active {
    background: var(--green) !important;
    box-shadow: 0 0 6px var(--green-glow) !important;
}

/* ── Regime banner accent line uses CSS var ───────────────────────── */
.regime-accent-line {
    position: absolute; top: 0; left: 0;
    height: 1px; width: 100%;
    background: var(--bar-grad, var(--line-2));
    opacity: 0.5;
}

/* ── Kill Streamlit's running-indicator spinner overlay on buttons ─── */
button[data-testid="baseButton-secondary"] .st-emotion-cache-ocqkz7,
button[data-testid="baseButton-primary"]   .st-emotion-cache-ocqkz7,
button[data-testid="baseButton-secondary"] > div > div,
button[data-testid="baseButton-primary"]   > div > div { display:none !important; }

/* ── Remove any default Streamlit focus ring across all buttons ─────── */
button:focus { outline: none !important; box-shadow: none !important; }
*:focus-visible { outline: none !important; box-shadow: none !important; }

/* ── Prevent the iframe overlay flash that appears on st.rerun ──────── */
[data-testid="stAppViewBlockContainer"] { will-change: auto !important; }
</style>
""", unsafe_allow_html=True)

# ── Instant button-press feedback injected into parent DOM ──────────────────
_components.html("""
<script>
(function attachCrispPress() {
  function press(e) {
    var b = e.currentTarget;
    b.style.transform = 'scale(0.97)';
    b.style.transition = 'transform 0.04s ease';
    setTimeout(function(){ b.style.transform = ''; }, 120);
  }
  function bindAll() {
    document.querySelectorAll(
      'button[data-testid="baseButton-secondary"], button[data-testid="baseButton-primary"]'
    ).forEach(function(b) {
      if (!b._crispBound) {
        b.addEventListener('mousedown', press);
        b._crispBound = true;
      }
    });
  }
  bindAll();
  var mo = new MutationObserver(bindAll);
  mo.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", height=0)


# ── Navigation Bar ──────────────────────────────────────────────────────────
_T_nav = get_theme()
_components.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  .nav-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px; height: 48px;
    background: {_T_nav['nav_bg']};
    border-bottom: 1px solid {_T_nav['nav_border']};
  }}
  .nav-wordmark {{ display: flex; align-items: center; gap: 12px; font-family: 'Barlow Condensed', sans-serif; font-size: 17px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; }}
  .nav-pip {{ width: 7px; height: 7px; border-radius: 50%; background: {_T_nav['nav_dot']}; animation: pulse 3s ease-in-out infinite; }}
  .nav-word   {{ color: {_T_nav['nav_t1']}; }}
  .nav-accent {{ color: {_T_nav['nav_t2']}; }}
  .nav-right  {{ display: flex; align-items: center; gap: 18px; }}
  .nav-status {{ display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: 2px; color: {_T_nav['nav_t2']}; text-transform: uppercase; }}
  .nav-dot {{ width: 5px; height: 5px; border-radius: 50%; background: {_T_nav['nav_dot']}; animation: pulse 2s ease-in-out infinite; }}
  .nav-divider {{ width: 1px; height: 16px; background: {_T_nav['nav_border']}; }}
  #live-clock {{ color: {_T_nav['nav_clock']}; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1px; }}
  #cdown-pill {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 1px; color: {_T_nav['nav_t2']}; border: 1px solid {_T_nav['nav_border']}; border-radius: 20px; padding: 3px 10px; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
</style>
<div class="nav-bar">
  <div class="nav-wordmark">
    <span class="nav-pip"></span>
    <span><span class="nav-word">CAPITANO</span><span class="nav-accent">TERMINAL</span></span>
  </div>
  <div class="nav-right">
    <div class="nav-status">
      <div class="nav-dot"></div>
      LIVE
    </div>
    <div class="nav-divider"></div>
    <span id="live-clock">--:--:--</span>
  </div>
</div>
<script>
  function updateClock() {{
    var now = new Date();
    var est = new Date(now.toLocaleString("en-US", {{timeZone: "America/New_York"}}));
    var h = String(est.getHours()).padStart(2,'0');
    var m = String(est.getMinutes()).padStart(2,'0');
    var s = String(est.getSeconds()).padStart(2,'0');
    document.getElementById('live-clock').textContent = h + ':' + m + ':' + s;
  }}
  updateClock(); setInterval(updateClock, 1000);
  var TOTAL = {AUTO_REFRESH_SECONDS}, secs = TOTAL;
  function tick() {{ secs = secs > 0 ? secs-1 : 0; var p = window.parent.document.getElementById('gex-cdown'); if(p) p.textContent = secs; }}
  setInterval(tick, 1000);
  function attachObserver() {{
    var signal = window.parent.document.getElementById('gex-refresh-signal');
    if (!signal) {{ setTimeout(attachObserver, 500); return; }}
    new MutationObserver(function() {{ secs = TOTAL; var p = window.parent.document.getElementById('gex-cdown'); if(p) p.textContent = secs; }}).observe(signal, {{ childList:true, characterData:true, subtree:true }});
  }}
  attachObserver();
</script>
""", height=50, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────────────────
def _d1d2(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return d1, d1 - sigma*math.sqrt(T)

def bs_price(S, K, T, r, q, sigma, flag):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    if flag == "C":
        return S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)

def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return math.exp(-q*T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))

def bs_delta(S, K, T, r, q, sigma, flag):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return math.exp(-q*T)*norm.cdf(d1) if flag=="C" else -math.exp(-q*T)*norm.cdf(-d1)

def bs_vega(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return S*math.exp(-q*T)*norm.pdf(d1)*math.sqrt(T)

def bs_charm(S, K, T, r, q, sigma, flag):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    c = -math.exp(-q*T)*norm.pdf(d1)*(2*(r-q)*T - d2*sigma*math.sqrt(T))/(2*T*sigma*math.sqrt(T))
    return c - q*math.exp(-q*T)*norm.cdf(d1) if flag=="C" else c + q*math.exp(-q*T)*norm.cdf(-d1)

def bs_vanna(S, K, T, r, q, sigma):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return -math.exp(-q*T)*norm.pdf(d1)*d2/sigma

def bs_vomma(S, K, T, r, q, sigma):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return bs_vega(S,K,T,r,q,sigma)*d1*d2/sigma

def bs_zomma(S, K, T, r, q, sigma):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return bs_gamma(S,K,T,r,q,sigma)*(d1*d2-1)/sigma

def implied_vol(market_price, S, K, T, r, q, flag):
    if T <= 0 or market_price <= 0: return np.nan
    intrinsic = max(0.0, (S-K) if flag=="C" else (K-S))
    if market_price <= intrinsic + 1e-4: return np.nan
    try:
        iv = brentq(lambda v: bs_price(S,K,T,r,q,v,flag) - market_price,
                    1e-5, 10.0, xtol=1e-6, maxiter=200)
        return iv if 0.005 < iv < 5.0 else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH — CBOE Public API (no key required)
# ─────────────────────────────────────────────────────────────────────────────
import requests as _requests

def _cboe_get(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.cboe.com/",
        "Origin": "https://www.cboe.com",
    }
    r = _requests.get(url, headers=headers, timeout=20)
    if not r.ok:
        raise RuntimeError(f"CBOE {url} returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()

# Yahoo Finance ticker mapping for real-time spot price
_YF_SYMBOL_MAP = {
    "SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT",
    "VIX": "^VIX",
}

def _fetch_spot_yahoo(ticker: str) -> float:
    """Fetch real-time spot price from Yahoo Finance.
    Returns real-time price during market hours (no delay).
    Raises on failure so caller can fall back to CBOE."""
    yf_sym = _YF_SYMBOL_MAP.get(ticker.upper(), ticker.upper())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
           f"?interval=1m&range=1d&includePrePost=true")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    r = _requests.get(url, headers=headers, timeout=8)
    if not r.ok:
        raise RuntimeError(f"Yahoo HTTP {r.status_code}")
    meta = r.json()["chart"]["result"][0]["meta"]
    # regularMarketPrice is real-time; previousClose is fallback
    price = meta.get("regularMarketPrice") or meta.get("previousClose")
    if not price or float(price) <= 0:
        raise RuntimeError("Yahoo returned no valid price")
    return float(price)

def get_spot(ticker: str) -> float:
    """Real-time spot price. Tries Yahoo Finance first (real-time),
    falls back to CBOE delayed quote on failure.
    Cache TTL = 10s so the 60s fragment refresh always gets a fresh print."""
    key = f"_spot_{ticker}"
    now = datetime.datetime.utcnow()
    cached = st.session_state.get(key)
    if cached and (now - cached["ts"]).total_seconds() < 10:
        return cached["val"]

    # ── Try Yahoo Finance (real-time) ──────────────────────────────────────
    try:
        val = _fetch_spot_yahoo(ticker)
        st.session_state[key] = {"val": val, "ts": now, "src": "live"}
        return val
    except Exception:
        pass

    # ── Fall back to CBOE (delayed ~15 min) ────────────────────────────────
    try:
        symbol = ticker.upper()
        data = _cboe_get(
            f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
        )
        val = float(data["data"]["current_price"])
        st.session_state[key] = {"val": val, "ts": now, "src": "delayed"}
        return val
    except Exception:
        pass

    # ── Last resort: return last known cached value even if stale ──────────
    if cached:
        return cached["val"]
    raise RuntimeError(f"Could not fetch spot price for {ticker}")


def _get_chain(ticker: str) -> dict:
    key = f"_chain_{ticker}"
    now = datetime.datetime.utcnow()
    cached = st.session_state.get(key)
    if cached and (now - cached["ts"]).total_seconds() < 58:
        return cached["data"]
    symbol = ticker.upper()
    for url in [
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json",
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json",
    ]:
        try:
            data = _cboe_get(url)
            if data.get("data", {}).get("options"):
                st.session_state[key] = {"data": data, "ts": now}
                return data
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch options chain for {ticker} from CBOE")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7: DYNAMIC ES / NQ CONVERSION RATIO
# ─────────────────────────────────────────────────────────────────────────────
if "es_spy_ratio" not in st.session_state:
    st.session_state.es_spy_ratio = None
if "nq_qqq_ratio" not in st.session_state:
    st.session_state.nq_qqq_ratio = None

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_yahoo_price(ticker: str) -> float:
    """Fetch a spot/futures price from Yahoo Finance query1 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    r = _requests.get(url, headers=headers, timeout=10)
    if not r.ok:
        raise RuntimeError(f"Yahoo {ticker} HTTP {r.status_code}")
    data = r.json()
    meta = data["chart"]["result"][0]["meta"]
    price = meta.get("regularMarketPrice") or meta.get("previousClose")
    if not price:
        raise RuntimeError(f"No price in Yahoo response for {ticker}")
    return float(price)

def get_es_spy_ratio(spy_spot: float) -> float:
    try:
        es_price = _fetch_yahoo_price("ES=F")
        ratio = es_price / spy_spot
        st.session_state.es_spy_ratio = ratio
        return ratio
    except Exception:
        pass
    if st.session_state.es_spy_ratio is not None:
        return st.session_state.es_spy_ratio
    try:
        spx_spot = get_spot("SPX")
        ratio = spx_spot / spy_spot
        st.session_state.es_spy_ratio = ratio
        return ratio
    except Exception:
        pass
    return 10.0

def get_nq_qqq_ratio(qqq_spot: float) -> float:
    try:
        nq_price = _fetch_yahoo_price("NQ=F")
        ratio = nq_price / qqq_spot
        st.session_state.nq_qqq_ratio = ratio
        return ratio
    except Exception:
        pass
    if st.session_state.nq_qqq_ratio is not None:
        return st.session_state.nq_qqq_ratio
    try:
        ndx_spot = get_spot("NDX")
        ratio = ndx_spot / qqq_spot
        st.session_state.nq_qqq_ratio = ratio
        return ratio
    except Exception:
        pass
    return 42.0


def _parse_cboe_chain(data, spot, max_expirations=4):
    options = data["data"].get("options", [])
    today   = datetime.date.today()

    def dte(e):
        return (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days

    def parse_symbol(sym):
        import re
        m = re.search(r'(\d{6})([CP])(\d{8})$', sym)
        if not m:
            return None, None, None
        date_str, flag, strike_str = m.group(1), m.group(2), m.group(3)
        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        strike = int(strike_str) / 1000.0
        return expiry, flag, strike

    by_exp = {}
    for opt in options:
        sym = opt.get("option", "")
        expiry, flag, strike = parse_symbol(sym)
        if not expiry or strike is None:
            continue
        by_exp.setdefault(expiry, []).append({**opt, "_expiry": expiry, "_flag": flag, "_strike": strike})

    sorted_exps = sorted(
        [e for e in by_exp if dte(e) >= 0],
        key=dte
    )[:max_expirations]

    result = {}
    for exp in sorted_exps:
        rows = []
        for opt in by_exp[exp]:
            rows.append({
                "strike":        opt["_strike"],
                "option_type":   opt["_flag"],
                "open_interest": float(opt.get("open_interest", 0) or 0),
                "volume":        float(opt.get("volume", 0) or 0),
                "bid":           float(opt.get("bid", 0) or 0),
                "ask":           float(opt.get("ask", 0) or 0),
                "iv":            float(opt.get("iv", 0) or 0),
            })
        if rows:
            result[exp] = pd.DataFrame(rows)
    return result, sorted_exps


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS CHAIN
# ─────────────────────────────────────────────────────────────────────────────
def _process_chain(chain_df, spot, T, r, q, exp, days):
    rows = []

    _atm_mask = (
        (chain_df["strike"] >= spot * 0.98) &
        (chain_df["strike"] <= spot * 1.02)
    )
    _atm_iv_vals = []
    for _, _row in chain_df[_atm_mask].iterrows():
        _iv_r = float(_row.get("iv", 0) or 0)
        if _iv_r > 0.05:
            _atm_iv_vals.append(_iv_r)
    _atm_iv_base = float(np.median(_atm_iv_vals)) if _atm_iv_vals else 0.20

    for _, row in chain_df.iterrows():
        flag = str(row["option_type"]) if "option_type" in row.index else "C"
        K    = float(row["strike"]        if "strike"        in row.index else 0)
        oi   = float(row["open_interest"] if "open_interest" in row.index else 0)
        vol  = float(row["volume"]        if "volume"        in row.index else 0)
        bid  = float(row["bid"]           if "bid"           in row.index else 0)
        ask  = float(row["ask"]           if "ask"           in row.index else 0)
        iv_r = float(row["iv"]            if "iv"            in row.index else 0)

        if K <= 0:
            continue

        dist_pct = abs(K - spot) / spot
        if dist_pct > 0.08:
            continue

        if oi < 100:
            continue

        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        else:
            mid = 0.0

        iv = np.nan

        # Priority 1: solve from live bid/ask mid (most accurate, requires real market)
        if mid > 0.05:
            iv = implied_vol(mid, spot, K, T, r, q, flag)

        # Priority 2: use CBOE's own IV field (available after-hours / when spread is wide)
        if (np.isnan(iv) or iv <= 0.005) and iv_r > 0.05:
            iv = float(iv_r)

        # Priority 3: smile extrapolation from ATM base (last resort, only if mid exists)
        if (np.isnan(iv) or iv <= 0.005) and mid > 0.05:
            iv = _atm_iv_base * (1.0 + dist_pct * 0.5)

        # No usable IV at all — skip this strike
        if np.isnan(iv) or iv <= 0.005:
            continue

        iv = min(iv, 1.5)

        gamma = bs_gamma(spot, K, T, r, q, iv)
        delta = bs_delta(spot, K, T, r, q, iv, flag)
        vega  = bs_vega(spot, K, T, r, q, iv)
        charm = bs_charm(spot, K, T, r, q, iv, flag)
        vanna = bs_vanna(spot, K, T, r, q, iv)
        vomma = bs_vomma(spot, K, T, r, q, iv)
        zomma = bs_zomma(spot, K, T, r, q, iv)

        gex_oi  = gamma * oi  * 100 * (spot ** 2) * 0.01 / 1e9
        gex_vol = gamma * vol * 100 * (spot ** 2) * 0.01 / 1e9

        rows.append({
            "strike": K, "expiry": exp, "dte": days, "flag": flag,
            "open_interest": oi, "volume": vol, "last_price": mid,
            "bid": bid, "ask": ask,
            "iv": iv, "delta": delta, "gamma": gamma,
            "vega": vega, "charm": charm, "vanna": vanna,
            "vomma": vomma, "zomma": zomma,
            "call_gex":     gex_oi  if flag == "C" else 0.0,
            "put_gex":     -gex_oi  if flag == "P" else 0.0,
            "call_vol_gex": gex_vol if flag == "C" else 0.0,
            "put_vol_gex": -gex_vol if flag == "P" else 0.0,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# FETCH OPTIONS DATA — HEATMAP (7 expirations, with 90 DTE cap)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_options_data_heatmap(ticker: str) -> tuple:
    today = datetime.date.today()
    def dte(e): return (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days
    raw_data = _get_chain(ticker)
    spot     = float(raw_data["data"]["current_price"])
    r, q     = RISK_FREE_RATE, DIV_YIELD.get(ticker, 0.01)
    chains, exps = _parse_cboe_chain(raw_data, spot, max_expirations=7)
    rows = []
    for exp in exps:
        days = dte(exp)
        if days > 90:
            continue
        T = max(days, 0.5) / 365.0
        rows.extend(_process_chain(chains[exp], spot, T, r, q, exp, days))
    if not rows:
        return pd.DataFrame(), spot, pd.DataFrame()
    return pd.DataFrame(), spot, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH OPTIONS DATA — MAIN (up to 4 expirations, 90 DTE cap)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_options_data(ticker: str, max_expirations: int = 4) -> tuple:
    today = datetime.date.today()
    def dte(e): return (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days
    raw_data = _get_chain(ticker)
    spot     = float(raw_data["data"]["current_price"])
    r, q     = RISK_FREE_RATE, DIV_YIELD.get(ticker, 0.01)
    chains, exps = _parse_cboe_chain(raw_data, spot, max_expirations=max_expirations)
    rows = []
    for exp in exps:
        days = dte(exp)
        if days > 90:
            continue
        T = max(days, 0.5) / 365.0
        rows.extend(_process_chain(chains[exp], spot, T, r, q, exp, days))
    if not rows:
        return pd.DataFrame(), spot, pd.DataFrame()

    raw        = pd.DataFrame(rows)
    call_vol_s = raw[raw["flag"]=="C"].groupby("strike")["volume"].sum().rename("volume_call")
    put_vol_s  = raw[raw["flag"]=="P"].groupby("strike")["volume"].sum().rename("volume_put")
    call_oi_s  = raw[raw["flag"]=="C"].groupby("strike")["open_interest"].sum().rename("call_oi")
    put_oi_s   = raw[raw["flag"]=="P"].groupby("strike")["open_interest"].sum().rename("put_oi")

    call_bid_s = raw[raw["flag"]=="C"].groupby("strike")["bid"].sum().rename("call_bid_sum")
    call_ask_s = raw[raw["flag"]=="C"].groupby("strike")["ask"].sum().rename("call_ask_sum")
    put_bid_s  = raw[raw["flag"]=="P"].groupby("strike")["bid"].sum().rename("put_bid_sum")
    put_ask_s  = raw[raw["flag"]=="P"].groupby("strike")["ask"].sum().rename("put_ask_sum")

    raw["_dex_val"] = raw["delta"] * raw["open_interest"] * 100
    raw["_vex_val"] = raw["vega"]  * raw["open_interest"] * 100 / 1e6
    raw["_cex_val"] = raw["charm"] * raw["open_interest"] * 100 / 1e6

    call_delta_s = raw[raw["flag"]=="C"].groupby("strike")["_dex_val"].sum().rename("call_dex")
    put_delta_s  = raw[raw["flag"]=="P"].groupby("strike")["_dex_val"].sum().rename("put_dex")
    call_vex_s   = raw[raw["flag"]=="C"].groupby("strike")["_vex_val"].sum().rename("call_vex")
    put_vex_s    = raw[raw["flag"]=="P"].groupby("strike")["_vex_val"].sum().rename("put_vex")
    call_cex_s   = raw[raw["flag"]=="C"].groupby("strike")["_cex_val"].sum().rename("call_cex")
    put_cex_s    = raw[raw["flag"]=="P"].groupby("strike")["_cex_val"].sum().rename("put_cex")

    agg = raw.groupby("strike").agg(
        call_gex=("call_gex","sum"), put_gex=("put_gex","sum"),
        call_vol_gex=("call_vol_gex","sum"), put_vol_gex=("put_vol_gex","sum"),
        vanna=("vanna","sum"), charm=("charm","sum"),
        vomma=("vomma","sum"), zomma=("zomma","sum"),
        vega=("vega","sum"), delta=("delta","sum"),
        open_interest=("open_interest","sum"),
        iv=("iv","mean"),
    ).reset_index()

    agg = (agg
           .join(call_vol_s,   on="strike")
           .join(put_vol_s,    on="strike")
           .join(call_oi_s,    on="strike")
           .join(put_oi_s,     on="strike")
           .join(call_bid_s,   on="strike")
           .join(call_ask_s,   on="strike")
           .join(put_bid_s,    on="strike")
           .join(put_ask_s,    on="strike")
           .join(call_delta_s, on="strike")
           .join(put_delta_s,  on="strike")
           .join(call_vex_s,   on="strike")
           .join(put_vex_s,    on="strike")
           .join(call_cex_s,   on="strike")
           .join(put_cex_s,    on="strike"))

    for c in ["volume_call","volume_put","call_oi","put_oi",
              "call_bid_sum","call_ask_sum","put_bid_sum","put_ask_sum",
              "call_dex","put_dex","call_vex","put_vex","call_cex","put_cex"]:
        agg[c] = agg[c].fillna(0)

    agg["gex_net"]     = agg["call_gex"]     + agg["put_gex"]
    agg["vol_gex_net"] = agg["call_vol_gex"] + agg["put_vol_gex"]
    agg["abs_gex"]     = agg["gex_net"].abs()
    agg["dex_net"]     = agg["call_dex"]     + agg["put_dex"]
    agg["vex_net"]     = agg["call_vex"]     + agg["put_vex"]
    agg["cex_net"]     = agg["call_cex"]     + agg["put_cex"]
    agg["dist_pct"]    = (agg["strike"] - spot) / spot * 100
    agg = agg.sort_values("strike").reset_index(drop=True)
    agg["velocity"]    = agg["gex_net"].diff().fillna(0)
    return agg, spot, raw


# ─────────────────────────────────────────────────────────────────────────────
# KEY LEVELS
# ─────────────────────────────────────────────────────────────────────────────
def compute_key_levels(df, spot, raw_df=None):
    df_s  = df.sort_values("strike")
    cum   = df_s["gex_net"].cumsum().values
    signs = cum[:-1] * cum[1:]
    flips = df_s["strike"].values[np.where(signs < 0)[0]]
    if len(flips):
        gamma_flip = float(flips[0])
    elif cum[-1] > 0:  # chain entirely net-positive → long gamma, flip is below all strikes
        gamma_flip = float(df_s["strike"].values[0]) * 0.99
    else:              # chain entirely net-negative → short gamma, flip is above all strikes
        gamma_flip = float(df_s["strike"].values[-1]) * 1.01

    pos = df[df["gex_net"] > 0]
    call_wall = float(pos.loc[pos["gex_net"].idxmax(), "strike"]) if not pos.empty else spot*1.01

    neg = df[df["gex_net"] < 0]
    put_wall = float(neg.loc[neg["gex_net"].idxmin(), "strike"]) if not neg.empty else spot*0.99

    if raw_df is not None and not raw_df.empty and "flag" in raw_df.columns:
        _call_oi = (raw_df[raw_df["flag"]=="C"]
                    .groupby("strike")["open_interest"].sum()
                    .reset_index().rename(columns={"open_interest":"call_oi"}))
        _put_oi  = (raw_df[raw_df["flag"]=="P"]
                    .groupby("strike")["open_interest"].sum()
                    .reset_index().rename(columns={"open_interest":"put_oi"}))
        mp_df = _call_oi.merge(_put_oi, on="strike", how="outer").fillna(0)
    elif "call_oi" in df_s.columns and "put_oi" in df_s.columns:
        mp_df = df_s[["strike", "call_oi", "put_oi"]].copy()
    else:
        mp_df = df_s[["strike"]].copy()
        mp_df["call_oi"] = 0.0
        mp_df["put_oi"]  = 0.0

    mp_df = mp_df[
        (mp_df["strike"] >= spot * 0.75) &
        (mp_df["strike"] <= spot * 1.25)
    ].reset_index(drop=True)

    if mp_df.empty or mp_df[["call_oi","put_oi"]].sum().sum() == 0:
        max_pain = spot
    else:
        strikes_mp  = mp_df["strike"].values
        call_oi_arr = mp_df["call_oi"].values
        put_oi_arr  = mp_df["put_oi"].values

        pain_values = []
        for i, k in enumerate(strikes_mp):
            mask_c    = strikes_mp < k
            call_pain = float(np.sum((k - strikes_mp[mask_c]) * call_oi_arr[mask_c])) * 100
            mask_p    = strikes_mp > k
            put_pain  = float(np.sum((strikes_mp[mask_p] - k) * put_oi_arr[mask_p])) * 100
            pain_values.append(call_pain + put_pain)

        max_pain = float(strikes_mp[int(np.argmin(pain_values))])

    return gamma_flip, call_wall, put_wall, max_pain


def compute_intraday_levels(df, spot):
    d = df.copy()
    d["abs_vol_gex"] = d["call_vol_gex"].abs() + d["put_vol_gex"].abs()
    vol_trigger = float(d.loc[d["abs_vol_gex"].idxmax(), "strike"]) \
                  if d["abs_vol_gex"].sum() > 0 else spot
    if d["vol_gex_net"].abs().sum() > 0:
        idx = d["vol_gex_net"].abs().idxmax()
        mom_wall = float(d.loc[idx, "strike"])
        mom_val  = float(d.loc[idx, "vol_gex_net"])
    else:
        mom_wall, mom_val = None, 0.0
    return vol_trigger, mom_wall, mom_val


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def build_heatmap_matrix(raw_df, spot, mode="oi"):
    if raw_df.empty:
        return None, None, None
    d = raw_df.copy()
    d = d[(d["strike"] >= spot*0.92) & (d["strike"] <= spot*1.08)]
    if d.empty:
        return None, None, None
    d["net_gex"] = (d["call_gex"] + d["put_gex"]) if mode=="oi" \
                   else (d["call_vol_gex"] + d["put_vol_gex"])
    piv = d.groupby(["strike","expiry"])["net_gex"].sum().reset_index()
    matrix = piv.pivot(index="strike", columns="expiry", values="net_gex").fillna(0)
    matrix = matrix.sort_index(ascending=True)
    return matrix.index.tolist(), matrix.columns.tolist(), matrix.values


# ─────────────────────────────────────────────────────────────────────────────
# IV-RV SPREAD
# ─────────────────────────────────────────────────────────────────────────────
def compute_iv_rv_spread(raw_df: pd.DataFrame, spot: float, ticker: str = "SPY") -> float:
    try:
        if raw_df.empty:
            return 0.0
        nearest_exp = raw_df["expiry"].min()
        near = raw_df[raw_df["expiry"] == nearest_exp].copy()
        atm  = near[(near["strike"] >= spot * 0.99) & (near["strike"] <= spot * 1.01)]
        if atm.empty:
            atm = near
        iv_atm = float(atm["iv"].mean()) * 100

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=30d"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = _requests.get(url, headers=headers, timeout=10)
        closes = []
        if r.ok:
            result = r.json()["chart"]["result"][0]
            closes = result["indicators"]["quote"][0].get("close", [])
            closes = [c for c in closes if c is not None]

        if len(closes) >= 5:
            log_rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
            hv20 = float(np.std(log_rets[-20:]) * math.sqrt(252) * 100)
        else:
            hv20 = 0.0

        return round(iv_atm - hv20, 2)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FLOW RATIO & NET FLOW
# ─────────────────────────────────────────────────────────────────────────────
def compute_flow(raw_df: pd.DataFrame, spot: float) -> tuple:
    """
    Flow Ratio using gamma-weighted dollar volume (call vs put).

    With CBOE snapshot data we have no tick-level trade prints — the stored
    last_price is the bid/ask mid, so any bid-ask aggressor classifier will
    always return 0.5.  The correct approach for snapshot data is to measure
    the *balance of dollar premium traded* between calls and puts, weighted
    by gamma so that near-ATM flow (where dealer hedging is most impactful)
    dominates.

      flow_ratio > 0.5  →  more gamma-weighted call volume  →  bullish flow
      flow_ratio < 0.5  →  more gamma-weighted put volume   →  bearish flow
      net_flow         =  call_gw_dollar_vol − put_gw_dollar_vol  ($)
    """
    if raw_df.empty:
        return 0.5, 0.0

    df = raw_df.copy()
    df["mid"] = ((df["bid"] + df["ask"]) / 2.0).clip(lower=0.01)

    # Dollar premium traded per contract × 100 shares
    df["dollar_vol"] = df["volume"] * df["mid"] * 100.0

    # Gamma weight: ATM options have highest gamma and drive the most
    # dealer hedging per dollar of volume, so they should dominate the signal.
    # Clip at zero — out-of-the-money far wings can have tiny negative gamma
    # artefacts from numerical BS that should not flip the direction.
    g = df["gamma"].clip(lower=0.0) if "gamma" in df.columns         else pd.Series(1.0, index=df.index)
    df["gw_vol"] = df["dollar_vol"] * g

    calls = df[df["flag"] == "C"]
    puts  = df[df["flag"] == "P"]

    call_flow = float(calls["gw_vol"].sum())
    put_flow  = float(puts["gw_vol"].sum())
    total     = call_flow + put_flow

    flow_ratio = call_flow / total if total > 0 else 0.5
    net_flow   = call_flow - put_flow   # positive = call-dominated flow

    return round(flow_ratio, 3), net_flow


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Asset selector + pip dial
# ─────────────────────────────────────────────────────────────────────────────
_TH_SEL = get_theme()
st.sidebar.markdown(f"""
<p class='sb-section'>Interface</p>
""", unsafe_allow_html=True)
_theme_names  = list(THEMES.keys())
_cur_theme    = st.session_state.get("ui_theme", "Default")
_chosen_theme = st.sidebar.selectbox(
    "Theme", _theme_names,
    index=_theme_names.index(_cur_theme) if _cur_theme in _theme_names else 0,
    key="theme_select", label_visibility="collapsed"
)
if _chosen_theme != st.session_state.get("ui_theme"):
    st.session_state.ui_theme = _chosen_theme
    st.rerun()
_sw_html = '<div style="display:flex;gap:5px;padding:8px 0 14px 0;">'
for _tn in _theme_names:
    _sw_accent = THEMES[_tn]["green"]
    _active_ring = f"box-shadow:0 0 0 2px {THEMES[_tn]['t2']};" if _tn == _cur_theme else "opacity:0.5;"
    _sw_html += f'<div title="{_tn}" style="width:12px;height:12px;border-radius:50%;background:{_sw_accent};{_active_ring}"></div>'
_sw_html += '</div>'
st.sidebar.markdown(_sw_html, unsafe_allow_html=True)

st.sidebar.markdown("<p class='sb-section'>Asset</p>", unsafe_allow_html=True)

_asset_options = ["SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLF", "XLE", "IBIT",
                  "SPX", "NDX", "RUT",
                  "AAPL", "NVDA", "TSLA", "AMZN", "MSFT", "META", "GOOGL"]
_cur = st.session_state.asset_choice if st.session_state.asset_choice in _asset_options else "SPY"
_chosen = st.sidebar.selectbox("Asset", _asset_options,
    index=_asset_options.index(_cur), key="asset_select",
    label_visibility="collapsed")
if _chosen != st.session_state.asset_choice:
    st.session_state.asset_choice = _chosen
    st.rerun()


def _get_equiv_config(ticker: str, spot: float):
    if ticker == "SPY":
        ratio = get_es_spy_ratio(spot)
        return "ES Equiv", ratio
    elif ticker == "SPX":
        return "ES Equiv", 1.0
    elif ticker == "QQQ":
        ratio = get_nq_qqq_ratio(spot)
        return "NQ Equiv", ratio
    elif ticker == "NDX":
        return "NQ Equiv", 1.0
    elif ticker == "IWM":
        try:
            rty_price = _fetch_yahoo_price("RTY=F")
            ratio = rty_price / spot
            return "RTY Equiv", ratio
        except Exception:
            return "RTY Equiv", 10.0
    elif ticker == "RUT":
        return "RTY Equiv", 1.0
    else:
        return f"{ticker} $Val", 1.0


# ── Expiration pip dial ─────────────────────────────────────────────────────
st.sidebar.markdown("<p class='sb-section'>Expirations</p>", unsafe_allow_html=True)

if "max_exp" not in st.session_state:
    st.session_state.max_exp = 1
max_exp = st.session_state.max_exp

pip_heights = [10, 14, 20, 28]
pips_html = '<div class="exp-dial-wrap">'
pips_html += f'<div class="exp-dial-header"><span class="exp-dial-label">Expirations</span><span class="exp-dial-value">{max_exp}</span></div>'
pips_html += '<div class="exp-pip-row">'
for _pi in range(1, 5):
    _active = "active" if _pi <= max_exp else "inactive"
    _h = pip_heights[_pi - 1]
    pips_html += f'<div class="exp-pip {_active}" style="height:{_h}px;"></div>'
pips_html += '</div></div>'
st.sidebar.markdown(pips_html, unsafe_allow_html=True)

_exp_cols = st.sidebar.columns(4)
for _i, _col in enumerate(_exp_cols, 1):
    with _col:
        if st.button(str(_i), key=f"exp_btn_{_i}",
                     type="primary" if _i == max_exp else "secondary"):
            st.session_state.max_exp = _i
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="none",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="JetBrains Mono, monospace", color=TEXT2, size=10),
    showlegend=False,
    hovermode="closest",
)

def add_reference_lines(fig, spot, gflip):
    _tr = get_theme()
    fig.add_hline(y=gflip, line_dash="dot", line_color=_tr["gamma_line"], line_width=1,
                  annotation_text="  Zero Γ", annotation_position="right",
                  annotation_font_color=_tr["chart_t3"], annotation_font_size=9)
    fig.add_hline(y=spot, line_dash="solid", line_color=_tr["spot_line"], line_width=1,
                  annotation_text="  Spot", annotation_position="right",
                  annotation_font_color=_tr["spot_line"], annotation_font_size=9)

def bar_layout(fig, x_series, x_title, spot):
    max_x = x_series.abs().quantile(0.97) if x_series.abs().sum() > 0 else 1.0
    fig.update_layout(
        **PLOTLY_BASE,
        height=700,
        bargap=0.14,
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=10, color=TEXT3)),
            range=[-max_x*1.35, max_x*1.35],
            gridcolor=LINE, gridwidth=1,
            zerolinecolor=LINE2, zerolinewidth=1,
            tickfont=dict(size=9, family="JetBrains Mono"),
        ),
        yaxis=dict(
            title=dict(text="Strike", font=dict(size=10, color=TEXT3)),
            range=[spot * 0.92, spot * 1.08],
            gridcolor=LINE, gridwidth=1,
            zerolinecolor=LINE2,
            tickfont=dict(size=9, family="JetBrains Mono"),
        ),
        margin=dict(t=10, r=100, b=50, l=60),
        hoverlabel=dict(
            bgcolor=_TC["chart_hover"],
            bordercolor=_TC["chart_hover_border"],
            font=dict(family="JetBrains Mono", size=10, color=_TC["chart_hover_text"]),
        ),
    )

def gex_bars(y, x, spacing, colors, customdata, hovertemplate):
    return go.Bar(
        y=y, x=x, orientation="h",
        width=spacing * 0.78,
        marker=dict(color=colors, line=dict(width=0), opacity=0.88),
        customdata=customdata,
        hovertemplate=hovertemplate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GEX LANDSCAPE — 3D Topographic
# ─────────────────────────────────────────────────────────────────────────────
def build_gex_landscape(df, spot_price):
    from scipy.ndimage import gaussian_filter

    topo = df[
        (df["strike"] >= spot_price * 0.95) &
        (df["strike"] <= spot_price * 1.05)
    ].sort_values("strike").copy()

    if topo.empty or len(topo) < 3:
        return None

    strikes = topo["strike"].values
    gex     = topo["gex_net"].values

    n_depth  = 40
    depth    = np.linspace(0, 1, n_depth)
    sigma_d  = 0.32
    envelope = np.exp(-0.5 * (depth / sigma_d) ** 2)

    Z = np.outer(gex, envelope)
    Z = gaussian_filter(Z, sigma=[1.4, 0.9])

    abs_max = max(float(np.abs(Z).max()), 1e-9)

    colorscale = get_theme()["surface_colorscale"]

    fig = go.Figure(go.Surface(
        x=strikes, y=depth, z=Z.T,
        colorscale=colorscale, cmin=-abs_max, cmax=abs_max,
        showscale=False, opacity=1.0,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False, size=abs_max*0.18)),
        hovertemplate="Strike: %{x:.0f}<br>Net GEX: %{z:.4f}B<extra></extra>",
        lighting=dict(ambient=0.55, diffuse=0.9, specular=0.15, roughness=0.65, fresnel=0.08),
        lightposition=dict(x=200, y=300, z=600),
    ))
    fig.add_trace(go.Scatter3d(
        x=[spot_price, spot_price], y=[0, 1], z=[0, 0],
        mode="lines", line=dict(color=get_theme()["spot_line"], width=2), hoverinfo="skip",
    ))
    fig.update_layout(
        height=340, paper_bgcolor=BG,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title="", showticklabels=True, tickfont=dict(size=7, color=TEXT3, family="JetBrains Mono"), showgrid=True, gridcolor=LINE, gridwidth=1, zeroline=False, backgroundcolor=BG, showspikes=False),
            yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False, backgroundcolor=BG, showspikes=False),
            zaxis=dict(title="", showticklabels=True, tickfont=dict(size=7, color=TEXT3, family="JetBrains Mono"), showgrid=True, gridcolor=LINE, zeroline=True, zerolinecolor=LINE2, zerolinewidth=2, backgroundcolor=BG, showspikes=False),
            camera=dict(eye=dict(x=1.4, y=-1.8, z=0.9), up=dict(x=0, y=0, z=1)),
            bgcolor=BG,
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=0.6, z=0.8),
        ),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KEY LEVELS PANEL HTML
# ─────────────────────────────────────────────────────────────────────────────
def render_kl_panel(spot, gflip, cwall, pwall, mpain, vtrig, mwall, mval):
    _tkl = get_theme()
    mc = _tkl["blue"]
    mw_display = f"${mwall:.2f}" if mwall else "—"

    rows = [
        (_tkl["t1"],    "Spot",       f"${spot:.2f}"),
        (_tkl["t2"],    "Zero Gamma", f"${gflip:.2f}"),
        (_tkl["green"], "Call Wall",  f"${cwall:.2f}"),
        (_tkl["red"],   "Put Wall",   f"${pwall:.2f}"),
        (_tkl["amber"], "Max Pain",      f"${mpain:.2f}"),
        (_tkl["amber"], "Vol Trigger",   f"${vtrig:.2f}"),
        (mc,            "Momentum Wall", mw_display),
    ]

    html = '<div class="kl-panel"><div class="kl-header">Key Levels</div>'
    for color, name, price in rows:
        html += (
            f'<div class="kl-row-full">'
            f'  <div class="kl-row-full-left">'
            f'    <span class="kl-dot" style="background:{color};box-shadow:0 0 6px {color}55;"></span>'
            f'    <span class="kl-name" style="color:{color}">{name}</span>'
            f'  </div>'
            f'  <span class="kl-price-val" style="color:{color}">{price}</span>'
            f'</div>'
        )
    html += '</div>'
    return html


# ─────────────────────────────────────────────────────────────────────────────
# DAILY LEVELS — Static swing-trader reference (cached 1 day / 1 week)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_daily_levels_0dte(ticker: str) -> dict:
    try:
        raw_data  = _get_chain(ticker)
        spot      = float(raw_data["data"]["current_price"])
        r, q      = RISK_FREE_RATE, DIV_YIELD.get(ticker, 0.01)
        chains, exps = _parse_cboe_chain(raw_data, spot, max_expirations=1)
        if not exps:
            return {}
        today  = datetime.date.today()
        exp    = exps[0]
        days   = (datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        T_     = max(days, 0.25) / 365.0
        rows   = _process_chain(chains[exp], spot, T_, r, q, exp, days)
        if not rows:
            return {}
        raw_df = pd.DataFrame(rows)
        agg_   = raw_df.groupby("strike").agg(
            call_gex=("call_gex","sum"), put_gex=("put_gex","sum"),
            gex_net=("call_gex","sum"),
            open_interest=("open_interest","sum"),
            volume=("volume","sum"),
        ).reset_index()
        agg_["gex_net"] = agg_["call_gex"] + agg_["put_gex"]

        pos    = agg_[agg_["gex_net"] > 0]
        neg    = agg_[agg_["gex_net"] < 0]
        c_wall = float(pos.loc[pos["gex_net"].idxmax(),"strike"]) if not pos.empty else spot*1.005
        p_wall = float(neg.loc[neg["gex_net"].idxmin(),"strike"]) if not neg.empty else spot*0.995

        ds = agg_.sort_values("strike")
        cum = ds["gex_net"].cumsum().values
        flips = ds["strike"].values[:-1][cum[:-1]*cum[1:]<0]
        gflip = float(flips[0]) if len(flips) else spot*0.99

        top_oi = agg_.nlargest(3, "open_interest")["strike"].tolist()

        calls_only = raw_df[raw_df["flag"] == "C"].copy()
        if not calls_only.empty:
            calls_only["_dist"] = (calls_only["strike"] - spot).abs()
            atm_row   = calls_only.nsmallest(1, "_dist")
            atm_iv    = float(atm_row["iv"].iloc[0])
            atm_iv    = min(atm_iv, 0.80)
        else:
            atm_iv = 0.18
        daily_move = spot * atm_iv / (252 ** 0.5)
        exp_hi  = round(spot + daily_move, 2)
        exp_lo  = round(spot - daily_move, 2)

        return dict(
            spot=spot, exp=exp, dte=days,
            call_wall=c_wall, put_wall=p_wall, gamma_flip=gflip,
            top_oi=top_oi, exp_hi=exp_hi, exp_lo=exp_lo,
            atm_iv=round(atm_iv*100, 2),
            total_gex=round(float(agg_["gex_net"].sum()), 4),
            fetched=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception as _e:
        return {"error": str(_e)}


@st.cache_data(ttl=86400 * 7, show_spinner=False)
def _fetch_weekly_levels(ticker: str) -> dict:
    try:
        raw_data  = _get_chain(ticker)
        spot      = float(raw_data["data"]["current_price"])
        r, q      = RISK_FREE_RATE, DIV_YIELD.get(ticker, 0.01)
        chains, exps = _parse_cboe_chain(raw_data, spot, max_expirations=4)
        today     = datetime.date.today()

        friday_exp = None
        for exp in exps:
            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            dte_     = (exp_date - today).days
            if exp_date.weekday() == 4 and dte_ <= 8:
                friday_exp = exp
                break
        if friday_exp is None and exps:
            for exp in exps:
                dte_ = (datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
                if dte_ <= 8:
                    friday_exp = exp
                    break
        if friday_exp is None:
            friday_exp = exps[0] if exps else None
        if friday_exp is None:
            return {}

        days  = (datetime.datetime.strptime(friday_exp, "%Y-%m-%d").date() - today).days
        T_    = max(days, 0.5) / 365.0
        chain_df_w = chains.get(friday_exp, pd.DataFrame())
        if chain_df_w.empty:
            return {}
        rows  = _process_chain(chain_df_w, spot, T_, r, q, friday_exp, days)
        if not rows:
            return {}
        raw_df = pd.DataFrame(rows)
        agg_   = raw_df.groupby("strike").agg(
            call_gex=("call_gex","sum"), put_gex=("put_gex","sum"),
            open_interest=("open_interest","sum"),
        ).reset_index()
        agg_["gex_net"] = agg_["call_gex"] + agg_["put_gex"]

        pos    = agg_[agg_["gex_net"] > 0]
        neg    = agg_[agg_["gex_net"] < 0]
        c_wall = float(pos.loc[pos["gex_net"].idxmax(),"strike"]) if not pos.empty else spot*1.01
        p_wall = float(neg.loc[neg["gex_net"].idxmin(),"strike"]) if not neg.empty else spot*0.99

        ds   = agg_.sort_values("strike")
        cum  = ds["gex_net"].cumsum().values
        flips = ds["strike"].values[:-1][cum[:-1]*cum[1:]<0]
        gflip = float(flips[0]) if len(flips) else spot*0.99

        strikes_mp  = agg_["strike"].values
        call_oi_arr = raw_df[raw_df["flag"]=="C"].groupby("strike")["open_interest"].sum().reindex(agg_["strike"]).fillna(0).values
        put_oi_arr  = raw_df[raw_df["flag"]=="P"].groupby("strike")["open_interest"].sum().reindex(agg_["strike"]).fillna(0).values
        pain_vals = []
        for k in strikes_mp:
            mask_c  = strikes_mp < k
            mask_p  = strikes_mp > k
            pain_vals.append(
                float(np.sum((k-strikes_mp[mask_c])*call_oi_arr[mask_c]))*100 +
                float(np.sum((strikes_mp[mask_p]-k)*put_oi_arr[mask_p]))*100
            )
        max_pain  = float(strikes_mp[int(np.argmin(pain_vals))]) if pain_vals else spot

        atm_ = raw_df[(raw_df["flag"]=="C") & raw_df["strike"].between(spot*0.99, spot*1.01)]
        atm_iv = float(atm_["iv"].mean()) if not atm_.empty else 0.18
        weekly_move = spot * atm_iv * (days/252)**0.5
        exp_hi = round(spot + weekly_move, 2)
        exp_lo = round(spot - weekly_move, 2)

        return dict(
            spot=spot, exp=friday_exp, dte=days,
            call_wall=c_wall, put_wall=p_wall, gamma_flip=gflip,
            max_pain=max_pain, exp_hi=exp_hi, exp_lo=exp_lo,
            atm_iv=round(atm_iv*100, 2),
            fetched=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )
    except Exception as _e:
        return {"error": str(_e)}


def _render_daily_levels(ticker: str, spot: float, df, raw_df, T: dict):
    def _level_card(label, value, color, sublabel=""):
        return f"""
        <div style="flex:1;min-width:120px;background:{T['bg1']};border:1px solid {T['line2']};
                    border-radius:4px;padding:12px 16px;border-top:2px solid {color};">
          <div style="font-family:'Barlow',sans-serif;font-size:8px;font-weight:600;
                      color:{T['t3']};letter-spacing:2px;text-transform:uppercase;
                      margin-bottom:6px;">{label}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:500;
                      color:{color};letter-spacing:-0.5px;">{value}</div>
          {f'<div style="font-family:Barlow,sans-serif;font-size:8px;color:{T["t3"]};margin-top:3px;">{sublabel}</div>' if sublabel else ''}
        </div>"""

    def _section_head(title, subtitle=""):
        return f"""
        <div style="display:flex;align-items:baseline;gap:12px;
                    padding:20px 0 10px 0;border-bottom:1px solid {T['line2']};
                    margin-bottom:14px;">
          <span style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
                       font-weight:700;letter-spacing:2px;text-transform:uppercase;
                       color:{T['t1']};">{title}</span>
          {f'<span style="font-family:Barlow,sans-serif;font-size:10px;color:{T["t3"]};">{subtitle}</span>' if subtitle else ''}
        </div>"""

    def _render_section(data: dict, section: str):
        if "error" in data:
            st.error(f"Could not load {section} levels: {data['error']}")
            return
        if not data:
            st.warning(f"No {section} data available.")
            return

        spot_v   = data.get("spot", spot)
        c_wall   = data.get("call_wall", spot_v)
        p_wall   = data.get("put_wall", spot_v)
        gflip    = data.get("gamma_flip", spot_v)
        exp_hi   = data.get("exp_hi", spot_v)
        exp_lo   = data.get("exp_lo", spot_v)
        atm_iv   = data.get("atm_iv", 0.0)
        exp_dt   = data.get("exp", "—")
        dte_v    = data.get("dte", 0)
        fetched  = data.get("fetched", "—")
        max_pain = data.get("max_pain", None)

        cards = ""
        cards += _level_card("Call Wall",    f"${c_wall:.2f}", T["green"],  f"Resistance · {exp_dt}")
        cards += _level_card("Put Wall",     f"${p_wall:.2f}", T["red"],    f"Support · {exp_dt}")
        cards += _level_card("Zero Gamma",   f"${gflip:.2f}",  T["t2"],    "Dealer flip line")
        if max_pain:
            cards += _level_card("Max Pain", f"${max_pain:.2f}", T["amber"], "Option pain target")
        st.markdown(f'<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;">{cards}</div>', unsafe_allow_html=True)

        range_cards = ""
        range_cards += _level_card("Expected High", f"${exp_hi:.2f}", T["green"], f"Spot + {((exp_hi/spot_v-1)*100):.1f}%")
        range_cards += _level_card("Expected Low",  f"${exp_lo:.2f}", T["red"],   f"Spot − {((1-exp_lo/spot_v)*100):.1f}%")
        range_cards += _level_card("ATM IV",         f"{atm_iv:.1f}%", T["amber"], f"{dte_v}DTE implied vol")
        top_oi = data.get("top_oi", [])
        if top_oi:
            oi_str = " · ".join([f"${s:.0f}" for s in top_oi[:3]])
            range_cards += _level_card("High OI Nodes", oi_str, T["violet"], "Top open interest strikes")
        st.markdown(f'<div style="display:flex;gap:8px;margin-bottom:6px;flex-wrap:wrap;">{range_cards}</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;color:{T["t3"]};padding:6px 0 14px 0;">Snapshot: {fetched} · Updates once per {"day" if dte_v <= 1 else "week"} · Not real-time</div>', unsafe_allow_html=True)

        range_span = max(abs(exp_hi - exp_lo), 1.0)
        pad        = range_span * 0.5
        x_min, x_max = exp_lo - pad, exp_hi + pad

        fig = go.Figure()
        fig.add_shape(type="rect", x0=exp_lo, x1=exp_hi, y0=0.2, y1=0.8,
                      fillcolor=T["green_glow"], line=dict(color=T["green"], width=1))
        for level, lbl, col in [
            (spot_v,  "SPOT",       T["t1"]),
            (c_wall,  "CALL WALL",  T["green"]),
            (p_wall,  "PUT WALL",   T["red"]),
            (gflip,   "ZERO Γ",     T["t2"]),
        ]:
            fig.add_shape(type="line", x0=level, x1=level, y0=0.05, y1=0.95,
                          line=dict(color=col, width=2 if lbl=="SPOT" else 1,
                                    dash="solid" if lbl=="SPOT" else "dot"))
            fig.add_annotation(x=level, y=1.05, text=f"<b>{lbl}</b><br>${level:.1f}",
                               showarrow=False, yref="y",
                               font=dict(size=8, color=col, family="JetBrains Mono"),
                               bgcolor=T["bg1"], borderpad=3)
        if max_pain:
            fig.add_shape(type="line", x0=max_pain, x1=max_pain, y0=0.05, y1=0.95,
                          line=dict(color=T["amber"], width=1, dash="dash"))
            fig.add_annotation(x=max_pain, y=0.0, text=f"MAX PAIN<br>${max_pain:.1f}",
                               showarrow=False, yref="y",
                               font=dict(size=8, color=T["amber"], family="JetBrains Mono"),
                               bgcolor=T["bg1"], borderpad=3)
        fig.update_layout(
            **PLOTLY_BASE,
            height=140,
            margin=dict(t=36, b=24, l=10, r=10),
            xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False,
                       tickfont=dict(size=9, family="JetBrains Mono", color=TEXT3),
                       showticklabels=True),
            yaxis=dict(visible=False, range=[-0.15, 1.2]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(_section_head("0DTE Levels", "Today's expiration · Intraday roadmap"), unsafe_allow_html=True)
        with st.spinner("Loading 0DTE levels…"):
            data_0dte = _fetch_daily_levels_0dte(ticker)
        _render_section(data_0dte, "0DTE")

    with col2:
        st.markdown(_section_head("Weekly Levels", "Friday expiration · Swing reference"), unsafe_allow_html=True)
        with st.spinner("Loading weekly levels…"):
            data_weekly = _fetch_weekly_levels(ticker)
        _render_section(data_weekly, "Weekly")

    st.markdown(f"""
    <div style="margin-top:20px;padding:12px 16px;background:{T['bg1']};
                border:1px solid {T['line']};border-radius:4px;
                border-left:3px solid {T['amber']};">
      <div style="font-family:'Barlow',sans-serif;font-size:10px;
                  font-weight:600;color:{T['amber']};letter-spacing:1.5px;
                  text-transform:uppercase;margin-bottom:4px;">Important Note</div>
      <div style="font-family:'Barlow',sans-serif;font-size:10px;color:{T['t2']};line-height:1.6;">
        Daily Levels are <b>static snapshots</b> computed once at page load and cached for the day/week.
        They show the GEX-derived support/resistance structure based on dealer positioning at the time of the snapshot.
        Levels update automatically each trading day (0DTE) and each week (Weekly).
        Not a buy/sell signal — always confirm with live price action.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# REPLAY — INTRADAY DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    yf_map = {"SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT"}
    yf_ticker = yf_map.get(ticker, ticker)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}?interval=1m&range=1d"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                   "Accept": "application/json"}
        r = _requests.get(url, headers=headers, timeout=15)
        if not r.ok:
            return pd.DataFrame()
        data = r.json()
        res  = data["chart"].get("result")
        if not res:
            return pd.DataFrame()
        res  = res[0]
        ts   = res.get("timestamp", [])
        q    = res["indicators"]["quote"][0]
        df   = pd.DataFrame({
            "ts":     pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York"),
            "open":   q.get("open",   []),
            "high":   q.get("high",   []),
            "low":    q.get("low",    []),
            "close":  q.get("close",  []),
            "volume": q.get("volume", []),
        })
        df = df.dropna(subset=["close", "open", "high", "low"])
        h, m = df["ts"].dt.hour, df["ts"].dt.minute
        df = df[~((h == 9) & (m < 30))]
        df = df[h.between(9, 15) | ((h == 16) & (m == 0))]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY — GEX SNAPSHOT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _compute_gex_snapshot_vec(strikes, ois, ivs, flags, spot, T, r, q):
    valid = (ivs > 0.005) & (ois >= 100)
    dist  = np.abs(strikes - spot) / spot
    valid = valid & (dist <= 0.10)
    if not valid.any():
        return pd.DataFrame(columns=["strike", "call_gex", "put_gex", "gex_net"])
    K, oi, iv, fl = strikes[valid], ois[valid], ivs[valid], flags[valid]
    T_  = max(float(T), 1e-9)
    sv  = np.maximum(iv * np.sqrt(T_), 1e-10)
    d1  = (np.log(spot / K) + (r - q + 0.5 * iv**2) * T_) / sv
    gam = np.exp(-q * T_) * np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * spot * sv)
    gam = np.minimum(gam, 5.0)
    gex = gam * oi * 100 * (spot**2) * 0.01 / 1e9
    out = pd.DataFrame({
        "strike":   K,
        "call_gex": np.where(fl == "C",  gex, 0.0),
        "put_gex":  np.where(fl == "P", -gex, 0.0),
    })
    agg = (out.groupby("strike")
             .agg(call_gex=("call_gex", "sum"), put_gex=("put_gex", "sum"))
             .reset_index())
    agg["gex_net"] = agg["call_gex"] + agg["put_gex"]
    return agg


def _precompute_replay_snapshots(raw_df: pd.DataFrame, df_intra: pd.DataFrame,
                                 ticker: str) -> list:
    r, q = RISK_FREE_RATE, DIV_YIELD.get(ticker, 0.01)
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    if "expiry" in raw_df.columns:
        dte0 = raw_df[raw_df["expiry"] == today_str]
        base = dte0 if not dte0.empty else raw_df
    else:
        base = raw_df

    strikes = base["strike"].values.astype(float)
    ois     = base["open_interest"].values.astype(float)
    ivs     = base["iv"].values.astype(float)
    flags   = base["flag"].values.astype(str)

    MO, MC = 9 * 60 + 30, 16 * 60
    TD     = MC - MO

    snapshots = []
    for _, row in df_intra.iterrows():
        ts   = row["ts"]
        spot = float(row["close"])
        mins_left = max(MC - (ts.hour * 60 + ts.minute), 1)
        T = max((mins_left / TD) / 252.0, 1.0 / (252 * TD))
        snapshots.append(
            _compute_gex_snapshot_vec(strikes, ois, ivs, flags, spot, T, r, q)
        )
    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY — RENDER
# ─────────────────────────────────────────────────────────────────────────────
def _render_replay_view(ticker, spot, T, gamma_flip, call_wall, put_wall,
                        max_pain, vol_trigger, df_gex, raw_df):
    import json as _json

    with st.spinner("Loading intraday data…"):
        df_intra = fetch_intraday_data(ticker)

    if df_intra.empty:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    padding:80px 20px;background:{T['bg1']};border:1px solid {T['line2']};
                    border-radius:6px;margin-top:10px;gap:12px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:700;
                      letter-spacing:3px;color:{T['t3']};text-transform:uppercase;">No Intraday Data</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{T['t3']};
                      letter-spacing:1px;text-align:center;line-height:2;">
            Try again during or after market hours (09:30–16:00 ET)
          </div>
        </div>""", unsafe_allow_html=True)
        return
    if len(df_intra) < 2:
        st.warning("Insufficient intraday data.")
        return

    snap_key = f"_replay_snaps_{ticker}_{datetime.date.today()}"
    if snap_key not in st.session_state or len(st.session_state[snap_key]) != len(df_intra):
        with st.spinner("Computing GEX snapshots… (once per session)"):
            st.session_state[snap_key] = _precompute_replay_snapshots(raw_df, df_intra, ticker)
    snaps = st.session_state[snap_key]

    cv, ctv = 0.0, 0.0
    vwap = []
    for _, r in df_intra.iterrows():
        tp = (float(r["high"]) + float(r["low"]) + float(r["close"])) / 3.0
        v  = max(float(r["volume"] or 0), 0.0)
        ctv += tp * v; cv += v
        vwap.append(round(ctv / cv if cv > 0 else tp, 2))

    gex_mag = [
        round(float(s["gex_net"].abs().sum()), 5) if (s is not None and not s.empty) else 0.0
        for s in snaps
    ]
    opening_mag = max(gex_mag[0] if gex_mag else 0.0, 1e-9)

    bars = []
    for i, (_, r) in enumerate(df_intra.iterrows()):
        bars.append({
            "t": r["ts"].strftime("%H:%M"),
            "o": round(float(r.get("open")  or 0), 2),
            "h": round(float(r.get("high")  or 0), 2),
            "l": round(float(r.get("low")   or 0), 2),
            "c": round(float(r.get("close") or 0), 2),
            "v": int(r.get("volume") or 0),
            "w": vwap[i],
        })

    snaps_js = []
    for s in snaps:
        if s is None or s.empty:
            snaps_js.append([])
        else:
            snaps_js.append([
                {"k": round(float(row["strike"]), 1), "g": round(float(row["gex_net"]), 5)}
                for _, row in s.iterrows()
            ])

    levels_d = {
        "gamma_flip":  round(gamma_flip, 2),
        "call_wall":   round(call_wall,  2),
        "put_wall":    round(put_wall,   2),
        "max_pain":    round(max_pain,   2),
        "vol_trigger": round(vol_trigger,2),
        "open_price":  round(float(df_intra.iloc[0]["open"]), 2),
        "ticker":      ticker,
        "date":        df_intra.iloc[0]["ts"].strftime("%b %d, %Y"),
    }
    theme_d = {
        "bg":    T["bg"],  "bg1":  T["bg1"],  "bg2": T["bg2"],
        "t1":    T["t1"],  "t2":   T["t2"],   "t3":  T["t3"],
        "line":  T["line"],"line2":T["line2"],
        "green": T["green"],"red": T["red"],  "amber":T["amber"],
        "blue":  T["blue"],"violet":T["violet"],
        "pos":   T["bar_pos"], "neg": T["bar_neg"],
    }

    css_vars = (
        f":root{{"
        f"--bg:{T['bg']};--bg1:{T['bg1']};--bg2:{T['bg2']};"
        f"--t1:{T['t1']};--t2:{T['t2']};--t3:{T['t3']};"
        f"--line:{T['line']};--line2:{T['line2']};"
        f"--green:{T['green']};--red:{T['red']};--amber:{T['amber']};"
        f"--violet:{T['violet']};"
        f"}}"
    )

    data_blk = (
        f"const BARS={_json.dumps(bars, separators=(',',':'))};"
        f"const SNAPS={_json.dumps(snaps_js, separators=(',',':'))};"
        f"const GEX_MAG={_json.dumps(gex_mag, separators=(',',':'))};"
        f"const OPENING_MAG={_json.dumps(opening_mag)};"
        f"const LEVELS={_json.dumps(levels_d)};"
        f"const TH={_json.dumps(theme_d)};"
    )

    html = _REPLAY_HTML.replace("/*CSS_VARS*/", css_vars, 1).replace("/*DATA*/", data_blk, 1)
    _components.html(html, height=730, scrolling=False)


# ─── HTML template ────────────────────────────────────────────────────────────
_REPLAY_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly-basic.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/*CSS_VARS*/
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;background:var(--bg);}
body{font-family:'JetBrains Mono',monospace;color:var(--t1);
     padding:8px;display:flex;flex-direction:column;gap:5px;}

/* ── Header ── */
#hdr{position:relative;overflow:hidden;display:flex;align-items:center;
     gap:14px;flex-wrap:wrap;padding:9px 14px;
     background:var(--bg1);border:1px solid var(--line2);border-radius:5px;flex-shrink:0;}
#prog{position:absolute;bottom:0;left:0;height:2px;width:0%;
      background:linear-gradient(90deg,var(--amber),transparent);
      border-radius:0 1px 0 0;}
.sep{width:1px;height:28px;background:var(--line2);flex-shrink:0;}
.hblk{display:flex;flex-direction:column;gap:1px;flex-shrink:0;}
.hlbl{font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--t3);}
#hprice{font-size:26px;font-weight:700;letter-spacing:-1px;color:var(--t1);line-height:1;}
#hpct{font-size:13px;font-weight:600;}
#htime{font-size:12px;font-weight:600;color:var(--amber);}
#hregime{font-size:14px;font-weight:700;letter-spacing:2px;font-family:'Barlow Condensed',sans-serif;}
#hgamma{font-size:14px;font-weight:700;}
#hhl{font-size:12px;}
#hzone{font-size:11px;font-weight:500;}

/* ── Charts row ── */
#charts{display:flex;gap:6px;flex:1;min-height:0;}
#gex-chart,#price-chart{min-width:0;flex:1;}
#gex-chart{flex:5;}
#price-chart{flex:7;}

/* ── Ramp ── */
#ramp{flex-shrink:0;height:68px;}

/* ── Controls ── */
#ctrls{display:flex;align-items:center;gap:5px;flex-shrink:0;padding:2px 0;}
.cb{background:transparent;border:1px solid var(--line2);color:var(--t3);
    font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;
    letter-spacing:.8px;text-transform:uppercase;border-radius:3px;
    padding:6px 11px;cursor:pointer;transition:border-color .08s,color .08s,background .08s;
    white-space:nowrap;user-select:none;}
.cb:hover{border-color:var(--t2);color:var(--t1);background:var(--bg2);}
.cb:active{transform:scale(.97);}
.cb.on{border-color:var(--t1);color:var(--t1);background:var(--bg2);}
#btn-play{min-width:95px;}
.sp{flex:1;}
#spd-lbl{font-size:9px;color:var(--t3);letter-spacing:1.5px;text-transform:uppercase;white-space:nowrap;}
#speed-sel{background:var(--bg1);border:1px solid var(--line2);color:var(--t2);
           font-family:'JetBrains Mono',monospace;font-size:10px;
           padding:5px 8px;border-radius:3px;cursor:pointer;outline:none;}

/* ── Scrubber ── */
#scrub-wrap{flex-shrink:0;padding:0 1px;}
#scrubber{width:100%;height:3px;-webkit-appearance:none;appearance:none;
          background:var(--line2);border-radius:2px;outline:none;cursor:pointer;display:block;}
#scrubber::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--amber);cursor:pointer;
  box-shadow:0 0 8px rgba(255,180,0,.5);}
#scrubber::-moz-range-thumb{width:14px;height:14px;border-radius:50%;
  background:var(--amber);border:none;cursor:pointer;}
.tlbl{display:flex;justify-content:space-between;font-size:8px;color:var(--t3);
      padding:3px 2px 0;letter-spacing:.5px;}
#tlbl-mid{color:var(--amber);font-weight:600;}
</style>
</head>
<body>

<!-- Header -->
<div id="hdr">
  <div id="prog"></div>
  <div style="display:flex;align-items:baseline;gap:8px;flex-shrink:0;">
    <span id="hprice">—</span>
    <span id="hpct">—</span>
  </div>
  <div class="sep"></div>
  <div class="hblk">
    <span class="hlbl" id="hticker-lbl">TICKER · DATE</span>
    <span id="htime">—</span>
  </div>
  <div class="sep"></div>
  <div class="hblk">
    <span class="hlbl">Session H / L</span>
    <span id="hhl">—</span>
  </div>
  <div class="sep"></div>
  <div class="hblk">
    <span class="hlbl">0DTE Γ Regime</span>
    <div style="display:flex;align-items:center;gap:6px;">
      <span id="hregime">—</span>
      <span id="hzone">—</span>
    </div>
  </div>
  <div class="sep"></div>
  <div class="hblk">
    <span class="hlbl">Gamma Intensity</span>
    <span id="hgamma">—</span>
  </div>
</div>

<!-- Charts -->
<div id="charts">
  <div id="gex-chart"></div>
  <div id="price-chart"></div>
</div>

<!-- Gamma Ramp -->
<div id="ramp"></div>

<!-- Controls -->
<div id="ctrls">
  <button class="cb" id="btn-open">⏮ Open</button>
  <button class="cb" id="btn-prev">◀ −1</button>
  <button class="cb" id="btn-play">▶  Play</button>
  <button class="cb" id="btn-next">+1 ▶</button>
  <button class="cb" id="btn-end">Close ⏭</button>
  <div class="sp"></div>
  <span id="spd-lbl">Speed</span>
  <select id="speed-sel">
    <option value="800">1×</option>
    <option value="400">2×</option>
    <option value="160" selected>5×</option>
    <option value="60">10×</option>
    <option value="25">20×</option>
  </select>
</div>

<!-- Scrubber -->
<div id="scrub-wrap">
  <input type="range" id="scrubber" min="0" max="1" value="0" step="1">
  <div class="tlbl">
    <span id="tlbl-lo">09:30 Open</span>
    <span id="tlbl-mid">—</span>
    <span id="tlbl-hi">16:00 Close</span>
  </div>
</div>

<script>
/*DATA*/

(function(){
  const s = document.documentElement.style;
  Object.entries(TH).forEach(([k,v]) => s.setProperty('--'+k.replace('_','-'), v));
})();

let idx = BARS.length - 1;
let playing = false;
let rafId = null;
let lastMs = 0;
let msPerBar = 160;
const MAX = BARS.length - 1;
const allT = BARS.map(b => b.t);
const allC = BARS.map(b => b.c);
const allW = BARS.map(b => b.w);

const gexDiv   = document.getElementById('gex-chart');
const priceDiv = document.getElementById('price-chart');
const rampDiv  = document.getElementById('ramp');
const scrubber = document.getElementById('scrubber');
const playBtn  = document.getElementById('btn-play');

function rgba(hex, a) {
  hex = hex.replace('#','');
  const r=parseInt(hex.slice(0,2),16), g=parseInt(hex.slice(2,4),16), b=parseInt(hex.slice(4,6),16);
  return 'rgba('+r+','+g+','+b+','+a+')';
}
function gfactor(i) { return (GEX_MAG[i]||0) / OPENING_MAG; }
function gfColor(gf) { return gf > 2.5 ? TH.red : gf > 1.2 ? TH.amber : TH.green; }

const allPrices = BARS.map(b => b.c);
const fixedLvls = [LEVELS.call_wall, LEVELS.put_wall, LEVELS.gamma_flip, LEVELS.max_pain];
const yAll = [...allPrices, ...fixedLvls];
const yPad = (Math.max(...yAll) - Math.min(...yAll)) * 0.04;
const Y_LO = Math.min(...yAll) - yPad;
const Y_HI = Math.max(...yAll) + yPad;

const CFG = {displayModeBar: false, responsive: true};
const BASE = {
  paper_bgcolor: TH.bg, plot_bgcolor: TH.bg,
  font: {family:'JetBrains Mono,monospace', color:TH.t2, size:10},
  showlegend: false,
  hoverlabel: {bgcolor:TH.bg1, bordercolor:TH.line2,
               font:{family:'JetBrains Mono', size:10, color:TH.t1}},
};

function initGEX() {
  Plotly.newPlot(gexDiv, [{
    type:'bar', orientation:'h',
    y:[], x:[], marker:{color:[], line:{width:0}},
    hovertemplate:'<b>Strike $%{y:.2f}</b><br>Net GEX: %{x:.5f}B<extra></extra>',
  }], {
    ...BASE,
    height: gexDiv.offsetHeight || 420,
    margin:{t:32,r:82,b:40,l:56},
    bargap: 0.10,
    xaxis:{gridcolor:TH.line, gridwidth:1, zerolinecolor:TH.line2, zerolinewidth:1,
           tickfont:{size:9,family:'JetBrains Mono'},
           title:{text:'Net GEX ($B)  ·  0DTE recalculated per bar',font:{size:9,color:TH.t3}}},
    yaxis:{gridcolor:TH.line, gridwidth:1, tickprefix:'$',
           tickfont:{size:9,family:'JetBrains Mono'},
           title:{text:'Strike',font:{size:9,color:TH.t3}}},
  }, CFG);
}

function initPrice() {
  const lvlShapes = [
    {type:'line',x0:0,x1:1,xref:'paper',y0:LEVELS.call_wall, y1:LEVELS.call_wall,line:{color:TH.green,width:1,dash:'dot'}},
    {type:'line',x0:0,x1:1,xref:'paper',y0:LEVELS.put_wall,  y1:LEVELS.put_wall, line:{color:TH.red,  width:1,dash:'dot'}},
    {type:'line',x0:0,x1:1,xref:'paper',y0:LEVELS.gamma_flip,y1:LEVELS.gamma_flip,line:{color:TH.t2, width:1,dash:'dash'}},
    {type:'line',x0:0,x1:1,xref:'paper',y0:LEVELS.max_pain,  y1:LEVELS.max_pain, line:{color:TH.amber,width:.9,dash:'dashdot'}},
    {type:'line',x0:allT[MAX],x1:allT[MAX],yref:'paper',y0:0,y1:1,line:{color:TH.amber,width:1.5}},
  ];
  const lvlAnnot = [
    {x:1,xref:'paper',y:LEVELS.call_wall, yref:'y',text:'  Call Wall $'+LEVELS.call_wall, showarrow:false,xanchor:'left',font:{size:8,color:TH.green,family:'JetBrains Mono'}},
    {x:1,xref:'paper',y:LEVELS.put_wall,  yref:'y',text:'  Put Wall $'+LEVELS.put_wall,   showarrow:false,xanchor:'left',font:{size:8,color:TH.red,  family:'JetBrains Mono'}},
    {x:1,xref:'paper',y:LEVELS.gamma_flip,yref:'y',text:'  Zero Γ $'+LEVELS.gamma_flip,   showarrow:false,xanchor:'left',font:{size:8,color:TH.t2,   family:'JetBrains Mono'}},
    {x:1,xref:'paper',y:LEVELS.max_pain,  yref:'y',text:'  Max Pain $'+LEVELS.max_pain,   showarrow:false,xanchor:'left',font:{size:8,color:TH.amber, family:'JetBrains Mono'}},
  ];
  Plotly.newPlot(priceDiv, [
    {type:'scatter',mode:'lines',x:allT,y:allC,line:{color:TH.t3,width:.8},opacity:.14,showlegend:false,hoverinfo:'skip'},
    {type:'scatter',mode:'lines',x:[],y:[],
     line:{color:TH.green,width:1.8},fill:'tozeroy',fillcolor:rgba(TH.green,.06),
     showlegend:false,hovertemplate:'%{x}  $%{y:.2f}<extra></extra>'},
    {type:'scatter',mode:'lines',x:[],y:[],
     line:{color:TH.violet,width:1.2},opacity:.7,showlegend:false,
     hovertemplate:'VWAP $%{y:.2f}<extra></extra>'},
    {type:'scatter',mode:'markers',x:[],y:[],
     marker:{color:TH.amber,size:8,line:{color:TH.bg,width:1.5}},
     showlegend:false,hoverinfo:'skip'},
  ], {
    ...BASE,
    height: priceDiv.offsetHeight || 420,
    margin:{t:32,r:128,b:40,l:56},
    title:{text:'Intraday Price  ·  '+LEVELS.ticker+'  ·  '+LEVELS.date,
           font:{size:10,color:TH.t3,family:'JetBrains Mono'},x:.01,xanchor:'left'},
    xaxis:{type:'category',gridcolor:TH.line,gridwidth:1,
           tickfont:{size:9,family:'JetBrains Mono',color:TH.t3},nticks:8},
    yaxis:{range:[Y_LO,Y_HI],gridcolor:TH.line,gridwidth:1,
           tickfont:{size:9,family:'JetBrains Mono',color:TH.t3},tickprefix:'$'},
    shapes: lvlShapes,
    annotations: lvlAnnot,
    hovermode:'x unified',
  }, CFG);
}

function initRamp() {
  Plotly.newPlot(rampDiv, [
    {type:'scatter',mode:'lines',fill:'tozeroy',
     x:GEX_MAG.map((_,i)=>i), y:GEX_MAG,
     line:{color:rgba(TH.amber,.25),width:1},
     fillcolor:rgba(TH.amber,.07),showlegend:false,hoverinfo:'skip'},
    {type:'scatter',mode:'lines',fill:'tozeroy',
     x:[], y:[],
     line:{color:TH.amber,width:2},
     fillcolor:rgba(TH.amber,.22),showlegend:false,hoverinfo:'skip'},
  ], {
    ...BASE, height:68,
    margin:{t:4,r:8,b:18,l:8},
    xaxis:{showgrid:false,zeroline:false,showticklabels:false},
    yaxis:{showgrid:false,zeroline:false,showticklabels:false},
    shapes:[{type:'line',x0:idx,x1:idx,yref:'paper',y0:0,y1:1,
              line:{color:TH.amber,width:1.5}}],
  }, CFG);
}

let lastGEXidx = -1;
function updateGEX() {
  if (lastGEXidx === idx) return;
  lastGEXidx = idx;

  const snap = SNAPS[idx] || [];
  const price = BARS[idx].c;
  const lo = price * 0.92, hi = price * 1.08;
  const filt = snap.filter(s => s.k >= lo && s.k <= hi);
  const gf = gfactor(idx);
  const op = Math.min(0.65 + 0.35 * Math.min(gf/3, 1), 1.0);
  const strikes = filt.map(s => s.k);
  const gexVals = filt.map(s => s.g);
  const maxX = Math.max(...gexVals.map(Math.abs), 0.0001);
  const spacing = strikes.length > 1
    ? (strikes[strikes.length-1] - strikes[0]) / (strikes.length - 1)
    : 1;
  const colors = gexVals.map(g => g>=0 ? rgba(TH.pos,op) : rgba(TH.neg,op));
  const gc = gfColor(gf);

  Plotly.react(gexDiv, [{
    type:'bar', orientation:'h',
    y:strikes, x:gexVals, width:spacing*.80,
    marker:{color:colors, line:{width:0}},
    hovertemplate:'<b>Strike $%{y:.2f}</b><br>Net GEX: %{x:.5f}B<extra></extra>',
  }], {
    ...BASE,
    height: gexDiv.offsetHeight || 420,
    margin:{t:32,r:82,b:40,l:56},
    bargap:0.10,
    title:{text:'GEX Landscape  ·  '+BARS[idx].t+'  ·  Γ×'+gf.toFixed(1)+' vs open',
           font:{size:10,color:TH.t3,family:'JetBrains Mono'},x:.01,xanchor:'left'},
    xaxis:{range:[-maxX*1.45,maxX*1.45],gridcolor:TH.line,gridwidth:1,
           zerolinecolor:TH.line2,zerolinewidth:1,tickfont:{size:9,family:'JetBrains Mono'},
           title:{text:'Net GEX ($B)  ·  0DTE recalculated per bar',font:{size:9,color:TH.t3}}},
    yaxis:{range:[price*.93,price*1.07],gridcolor:TH.line,gridwidth:1,
           tickprefix:'$',tickfont:{size:9,family:'JetBrains Mono'},
           title:{text:'Strike',font:{size:9,color:TH.t3}}},
    shapes:[
      {type:'line',x0:-maxX*1.5,x1:maxX*1.5,y0:LEVELS.gamma_flip,y1:LEVELS.gamma_flip,
       line:{color:TH.t3,width:1,dash:'dot'}},
      {type:'line',x0:-maxX*1.5,x1:maxX*1.5,y0:price,y1:price,
       line:{color:TH.amber,width:2.5}},
    ],
    annotations:[
      {x:maxX*1.38,y:price,text:'<b>$'+price.toFixed(2)+'</b>',
       showarrow:false,font:{size:10,color:TH.bg,family:'JetBrains Mono'},
       bgcolor:TH.amber,borderpad:4,xanchor:'right'},
      {x:-maxX*1.38,y:price*1.05,text:'<b>Γ×'+gf.toFixed(1)+'</b>  '+BARS[idx].t,
       showarrow:false,xanchor:'left',
       font:{size:9,color:gc,family:'JetBrains Mono'},bgcolor:TH.bg1,borderpad:3},
    ],
    hoverlabel:{bgcolor:TH.bg1,bordercolor:TH.line2,font:{family:'JetBrains Mono',size:10,color:TH.t1}},
  });
}

let prevDir = null;
function updatePrice() {
  const price = BARS[idx].c;
  const dir = price >= LEVELS.open_price ? 'up' : 'dn';
  const lc  = dir==='up' ? TH.green : TH.red;
  const fc  = rgba(dir==='up' ? TH.green : TH.red, .06);

  const px = allT.slice(0, idx+1);
  const pc = allC.slice(0, idx+1);
  const pw = allW.slice(0, idx+1);

  Plotly.restyle(priceDiv, {x:[px,px], y:[pc,pw]}, [1,2]);
  Plotly.restyle(priceDiv, {x:[[allT[idx]]], y:[[price]]}, [3]);
  if (dir !== prevDir) {
    Plotly.restyle(priceDiv, {'line.color':lc, fillcolor:fc}, [1]);
    prevDir = dir;
  }
  Plotly.relayout(priceDiv, {'shapes[4].x0':allT[idx],'shapes[4].x1':allT[idx]});
}

function updateRamp() {
  const px = GEX_MAG.slice(0,idx+1).map((_,i)=>i);
  const py = GEX_MAG.slice(0,idx+1);
  Plotly.restyle(rampDiv, {x:[px], y:[py]}, [1]);
  Plotly.relayout(rampDiv, {'shapes[0].x0':idx,'shapes[0].x1':idx});
}

function updateHeader() {
  const b = BARS[idx];
  const price = b.c;
  const pct = (price - LEVELS.open_price) / LEVELS.open_price * 100;
  const pctSign = pct>=0?'+':'';
  const pctColor = pct>=0 ? TH.green : TH.red;
  const gf = gfactor(idx);
  const gc = gfColor(gf);
  const isLong = price > LEVELS.gamma_flip;
  const rc = isLong ? TH.green : TH.red;

  const sl = BARS.slice(0,idx+1);
  const sessHi = Math.max(...sl.map(x=>x.h));
  const sessLo = Math.min(...sl.map(x=>x.l));

  const zone = price > LEVELS.call_wall ? 'ABOVE CALL WALL'
              : price < LEVELS.put_wall  ? 'BELOW PUT WALL'
              : 'IN RANGE';
  const zc = (price > LEVELS.call_wall || price < LEVELS.put_wall) ? TH.red : TH.amber;

  document.getElementById('hprice').textContent = '$'+price.toFixed(2);
  const pctEl = document.getElementById('hpct');
  pctEl.textContent = pctSign+pct.toFixed(2)+'%';
  pctEl.style.color = pctColor;
  document.getElementById('htime').textContent = '⏱ '+b.t+'  '+(idx+1)+'/'+(MAX+1);
  document.getElementById('hticker-lbl').textContent = LEVELS.ticker+' · '+LEVELS.date;
  const hlEl = document.getElementById('hhl');
  hlEl.innerHTML = '<span style="color:'+TH.green+'">$'+sessHi.toFixed(2)+'</span>'
                 + ' / <span style="color:'+TH.red+'">$'+sessLo.toFixed(2)+'</span>';
  const rEl = document.getElementById('hregime');
  rEl.textContent = isLong?'LONG Γ':'SHORT Γ'; rEl.style.color = rc;
  const zEl = document.getElementById('hzone');
  zEl.textContent = zone; zEl.style.color = zc;
  const gEl = document.getElementById('hgamma');
  gEl.textContent = 'Γ×'+gf.toFixed(1); gEl.style.color = gc;

  document.getElementById('prog').style.width = (idx/MAX*100).toFixed(1)+'%';
  scrubber.value = idx;
  document.getElementById('tlbl-mid').textContent = '⏱ '+b.t;
}

function updateAll() {
  updateGEX();
  updatePrice();
  updateRamp();
  updateHeader();
}

function animLoop(ts) {
  if (!playing) return;
  if (ts - lastMs >= msPerBar) {
    if (idx < MAX) {
      idx++;
      updateAll();
      lastMs = ts;
    } else {
      stopPlay();
      return;
    }
  }
  rafId = requestAnimationFrame(animLoop);
}

function startPlay() {
  if (idx >= MAX) { idx = 0; updateAll(); }
  playing = true;
  playBtn.textContent = '⏸  Pause';
  playBtn.classList.add('on');
  lastMs = 0;
  rafId = requestAnimationFrame(animLoop);
}

function stopPlay() {
  playing = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  playBtn.textContent = '▶  Play';
  playBtn.classList.remove('on');
}

document.getElementById('btn-open').addEventListener('click', function() {
  stopPlay(); idx = 0; updateAll();
});
document.getElementById('btn-prev').addEventListener('click', function() {
  stopPlay(); idx = Math.max(0, idx-1); updateAll();
});
playBtn.addEventListener('click', function() {
  if (playing) stopPlay(); else startPlay();
});
document.getElementById('btn-next').addEventListener('click', function() {
  stopPlay(); idx = Math.min(MAX, idx+1); updateAll();
});
document.getElementById('btn-end').addEventListener('click', function() {
  stopPlay(); idx = MAX; updateAll();
});
scrubber.addEventListener('input', function() {
  stopPlay(); idx = parseInt(this.value); updateAll();
});
document.getElementById('speed-sel').addEventListener('change', function() {
  msPerBar = parseInt(this.value);
});

window.addEventListener('load', function() {
  scrubber.max = MAX;
  scrubber.value = idx;
  if (BARS.length > 0) {
    document.getElementById('tlbl-lo').textContent = BARS[0].t+' Open';
    document.getElementById('tlbl-hi').textContent = BARS[MAX].t+' Close';
  }
  initGEX();
  initPrice();
  initRamp();
  updateAll();
});
</script>
</body>
</html>"""
# ─────────────────────────────────────────────────────────────────────────────
# GITHUB REPLAY STORAGE
# ─────────────────────────────────────────────────────────────────────────────
import json as _json
import base64 as _base64

def _gh_headers():
    try:
        token = st.secrets["github"]["token"]
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    except Exception:
        return {}

def _gh_repo():
    try:
        return st.secrets["github"]["repo"]
    except Exception:
        return ""

def _gh_branch():
    try:
        return st.secrets["github"].get("branch", "main")
    except Exception:
        return "main"

def _gh_api(path: str) -> str:
    return f"https://api.github.com/repos/{_gh_repo()}/contents/{path}"

def _gh_read(path: str) -> tuple:
    """Returns (content_dict_or_None, sha_or_None)"""
    r = _requests.get(_gh_api(path), headers=_gh_headers(), timeout=15)
    if r.status_code == 404:
        return None, None
    if not r.ok:
        return None, None
    data = r.json()
    raw  = _base64.b64decode(data["content"]).decode("utf-8")
    return _json.loads(raw), data["sha"]

def _gh_write(path: str, content: dict, message: str, sha=None) -> dict:
    """Write JSON to GitHub. Pass sha to update existing file."""
    encoded = _base64.b64encode(
        _json.dumps(content, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")
    body = {
        "message": message,
        "content": encoded,
        "branch":  _gh_branch(),
    }
    if sha:
        body["sha"] = sha
    r = _requests.put(_gh_api(path), headers=_gh_headers(),
                      json=body, timeout=30)
    if r.ok:
        return {"ok": True, "msg": f"Written to {path}"}
    return {"ok": False, "err": f"HTTP {r.status_code}: {r.text[:200]}"}


def _github_save_replay(ticker: str, date_str: str, T: dict) -> dict:
    """
    Saves today's intraday bars + precomputed GEX snapshots to GitHub at:
      replays/index.json          — manifest of all saved replays
      replays/{ticker}_{date}.json — the actual replay data
    """
    if not _gh_repo():
        return {"ok": False, "err": "No GitHub repo configured in st.secrets"}
    if not _gh_headers():
        return {"ok": False, "err": "No GitHub token configured in st.secrets"}

    # ── Fetch intraday bars ────────────────────────────────────────────────
    df_intra = fetch_intraday_data(ticker)
    if df_intra.empty:
        return {"ok": False, "err": "No intraday data available to save"}

    # ── Fetch current options chain for GEX snapshots ─────────────────────
    try:
        _, _, raw_df = fetch_options_data(ticker, max_expirations=4)
    except Exception as e:
        return {"ok": False, "err": f"Could not fetch options chain: {e}"}

    if raw_df.empty:
        return {"ok": False, "err": "Options chain returned no data"}

    # ── Compute GEX snapshots for every bar ───────────────────────────────
    snaps = _precompute_replay_snapshots(raw_df, df_intra, ticker)

    # ── Compute key levels for the saved replay ───────────────────────────
    try:
        spot = float(df_intra["close"].iloc[-1])
        agg, _, _raw = fetch_options_data(ticker, max_expirations=1)
        gflip, cwall, pwall, mpain = compute_key_levels(agg, spot, _raw) \
            if not agg.empty else (spot, spot, spot, spot)
        vtrig, mwall, _ = compute_intraday_levels(agg, spot) \
            if not agg.empty else (spot, spot, 0.0)
    except Exception:
        spot  = float(df_intra["close"].iloc[-1])
        gflip = cwall = pwall = mpain = vtrig = mwall = spot

    # ── Serialise bars ─────────────────────────────────────────────────────
    bars_out = []
    for _, row in df_intra.iterrows():
        bars_out.append({
            "t": row["ts"].strftime("%H:%M"),
            "o": round(float(row.get("open")  or 0), 2),
            "h": round(float(row.get("high")  or 0), 2),
            "l": round(float(row.get("low")   or 0), 2),
            "c": round(float(row.get("close") or 0), 2),
            "v": int(row.get("volume") or 0),
        })

    # ── Serialise GEX snapshots ────────────────────────────────────────────
    snaps_out = []
    for s in snaps:
        if s is None or s.empty:
            snaps_out.append([])
        else:
            snaps_out.append([
                {"k": round(float(r["strike"]), 1),
                 "g": round(float(r["gex_net"]), 6)}
                for _, r in s.iterrows()
            ])

    # ── GEX magnitude per bar (for ramp) ──────────────────────────────────
    gex_mag = [
        round(float(s["gex_net"].abs().sum()), 6)
        if (s is not None and not s.empty) else 0.0
        for s in snaps
    ]
    opening_mag = max(gex_mag[0] if gex_mag else 0.0, 1e-9)

    replay_doc = {
        "ticker":      ticker,
        "date":        date_str,
        "saved_at":    datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "spot":        round(spot, 2),
        "levels": {
            "gamma_flip":  round(gflip, 2),
            "call_wall":   round(cwall, 2),
            "put_wall":    round(pwall, 2),
            "max_pain":    round(mpain, 2),
            "vol_trigger": round(vtrig, 2),
            "open_price":  round(float(df_intra.iloc[0]["open"]), 2),
        },
        "bars":         bars_out,
        "snaps":        snaps_out,
        "gex_mag":      gex_mag,
        "opening_mag":  opening_mag,
    }

    # ── Write replay file ──────────────────────────────────────────────────
    file_path = f"replays/{ticker}_{date_str}.json"
    _, existing_sha = _gh_read(file_path)
    write_result = _gh_write(
        path    = file_path,
        content = replay_doc,
        message = f"Save replay: {ticker} {date_str}",
        sha     = existing_sha,
    )
    if not write_result.get("ok"):
        return write_result

    # ── Update index ───────────────────────────────────────────────────────
    index, index_sha = _gh_read("replays/index.json")
    if index is None:
        index = {"replays": []}

    # Remove old entry for same ticker+date if exists
    index["replays"] = [
        x for x in index["replays"]
        if not (x["ticker"] == ticker and x["date"] == date_str)
    ]
    index["replays"].append({
        "ticker":   ticker,
        "date":     date_str,
        "saved_at": replay_doc["saved_at"],
        "bars":     len(bars_out),
        "file":     file_path,
    })
    # Keep most recent 100 replays
    index["replays"] = sorted(
        index["replays"], key=lambda x: x["date"], reverse=True
    )[:100]

    _gh_write(
        path    = "replays/index.json",
        content = index,
        message = f"Update replay index: {ticker} {date_str}",
        sha     = index_sha,
    )

    return {"ok": True, "msg": f"{ticker} {date_str} — {len(bars_out)} bars saved"}


# ─────────────────────────────────────────────────────────────────────────────
# SAVED REPLAYS VIEWER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _gh_load_index() -> list:
    index, _ = _gh_read("replays/index.json")
    if index is None:
        return []
    return index.get("replays", [])

@st.cache_data(ttl=3600, show_spinner=False)
def _gh_load_replay(file_path: str) -> dict:
    data, _ = _gh_read(file_path)
    return data or {}


def _render_saved_replays(T: dict):

    st.markdown('<div class="sec-head">Saved Replays</div>', unsafe_allow_html=True)

    # ── Check secrets ──────────────────────────────────────────────────────
    if not _gh_repo():
        st.markdown(f"""
        <div style="background:{T['bg1']};border:1px solid {T['amber']};
                    border-left:3px solid {T['amber']};border-radius:4px;
                    padding:16px 20px;margin:8px 0;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
                      font-weight:700;letter-spacing:2px;color:{T['amber']};
                      text-transform:uppercase;margin-bottom:8px;">GitHub Not Configured</div>
          <div style="font-family:'Barlow',sans-serif;font-size:11px;
                      color:{T['t2']};line-height:1.8;">
            Add these to your Streamlit secrets:<br><br>
            <code style="background:{T['bg2']};padding:8px 12px;border-radius:3px;
                         display:block;font-size:10px;color:{T['green']};">
            [github]<br>
            token  = "ghp_yourtoken"<br>
            repo   = "yourusername/yourrepo"<br>
            branch = "main"
            </code>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load index ─────────────────────────────────────────────────────────
    with st.spinner("Loading saved replays…"):
        replays = _gh_load_index()

    if not replays:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:80px 20px;
                    background:{T['bg1']};border:1px solid {T['line2']};
                    border-radius:6px;margin-top:10px;gap:12px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;
                      font-weight:700;letter-spacing:3px;color:{T['t3']};
                      text-transform:uppercase;">No Saved Replays Yet</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                      color:{T['t3']};letter-spacing:1px;text-align:center;
                      line-height:2;">
            Go to the ⏱ Replay tab during market hours<br>
            and hit 💾 Save to store a session here.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Index table ────────────────────────────────────────────────────────
    st.markdown(f'<div class="sub-head">Saved Sessions  ·  {len(replays)} replays</div>',
                unsafe_allow_html=True)

    # Group by ticker for filter
    tickers_saved = sorted(set(r["ticker"] for r in replays))
    _filter_cols  = st.columns([2, 6])
    with _filter_cols[0]:
        _ticker_filter = st.selectbox(
            "Filter by ticker", ["All"] + tickers_saved,
            key="saves_ticker_filter", label_visibility="collapsed"
        )

    filtered = replays if _ticker_filter == "All" \
               else [r for r in replays if r["ticker"] == _ticker_filter]

    # Render replay cards
    for entry in filtered:
        _date    = entry["date"]
        _ticker  = entry["ticker"]
        _bars    = entry.get("bars", "?")
        _saved   = entry.get("saved_at", "")
        _file    = entry["file"]

        _card_key = f"load_{_ticker}_{_date}"
        _col1, _col2, _col3, _col4 = st.columns([2, 2, 2, 1])

        with _col1:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:13px;'
                f'font-weight:600;color:{T["t1"]};padding-top:8px;">'
                f'{_ticker} &nbsp; <span style="color:{T["amber"]}">{_date}</span></div>',
                unsafe_allow_html=True
            )
        with _col2:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;'
                f'color:{T["t3"]};padding-top:10px;">{_bars} bars &nbsp;·&nbsp; {_saved}</div>',
                unsafe_allow_html=True
            )
        with _col3:
            pass
        with _col4:
            if st.button("▶ Load", key=_card_key, use_container_width=True):
                st.session_state["saved_replay_file"]   = _file
                st.session_state["saved_replay_ticker"] = _ticker
                st.session_state["saved_replay_date"]   = _date
                st.rerun()

        st.markdown(
            f'<div style="height:1px;background:{T["line"]};margin:2px 0 6px 0;"></div>',
            unsafe_allow_html=True
        )

    # ── Replay player ──────────────────────────────────────────────────────
    if "saved_replay_file" not in st.session_state:
        return

    _active_file   = st.session_state["saved_replay_file"]
    _active_ticker = st.session_state["saved_replay_ticker"]
    _active_date   = st.session_state["saved_replay_date"]

    st.markdown(
        f'<div class="sub-head">Replaying: {_active_ticker} · {_active_date}</div>',
        unsafe_allow_html=True
    )

    with st.spinner(f"Loading {_active_ticker} {_active_date}…"):
        replay_data = _gh_load_replay(_active_file)

    if not replay_data:
        st.error("Could not load replay data from GitHub.")
        return

    bars        = replay_data.get("bars", [])
    snaps_raw   = replay_data.get("snaps", [])
    gex_mag     = replay_data.get("gex_mag", [])
    opening_mag = replay_data.get("opening_mag", 1.0)
    levels      = replay_data.get("levels", {})
    levels["ticker"] = _active_ticker
    levels["date"]   = _active_date

    if not bars:
        st.warning("No bar data in this replay.")
        return

    # Pass directly into the existing JS replay widget
    import json as _json2
    css_vars = (
        f":root{{"
        f"--bg:{T['bg']};--bg1:{T['bg1']};--bg2:{T['bg2']};"
        f"--t1:{T['t1']};--t2:{T['t2']};--t3:{T['t3']};"
        f"--line:{T['line']};--line2:{T['line2']};"
        f"--green:{T['green']};--red:{T['red']};--amber:{T['amber']};"
        f"--violet:{T['violet']};"
        f"}}"
    )
    theme_d = {
        "bg": T["bg"], "bg1": T["bg1"], "bg2": T["bg2"],
        "t1": T["t1"], "t2": T["t2"],  "t3": T["t3"],
        "line": T["line"], "line2": T["line2"],
        "green": T["green"], "red": T["red"], "amber": T["amber"],
        "blue": T["blue"], "violet": T["violet"],
        "pos": T["bar_pos"], "neg": T["bar_neg"],
    }
    data_blk = (
        f"const BARS={_json2.dumps(bars, separators=(',',':'))};"
        f"const SNAPS={_json2.dumps(snaps_raw, separators=(',',':'))};"
        f"const GEX_MAG={_json2.dumps(gex_mag, separators=(',',':'))};"
        f"const OPENING_MAG={_json2.dumps(opening_mag)};"
        f"const LEVELS={_json2.dumps(levels, separators=(',',':'))};"
        f"const TH={_json2.dumps(theme_d, separators=(',',':'))};"
    )

    html = _REPLAY_HTML.replace("/*CSS_VARS*/", css_vars, 1).replace("/*DATA*/", data_blk, 1)
    _components.html(html, height=730, scrolling=False)

    # Delete button
    if st.button(f"🗑 Delete this replay", key="delete_replay_btn"):
        with st.spinner("Deleting…"):
            _, sha = _gh_read(_active_file)
            if sha:
                _requests.delete(
                    _gh_api(_active_file),
                    headers=_gh_headers(),
                    json={
                        "message": f"Delete replay {_active_ticker} {_active_date}",
                        "sha": sha,
                        "branch": _gh_branch(),
                    },
                    timeout=15,
                )
            # Remove from index
            index, index_sha = _gh_read("replays/index.json")
            if index:
                index["replays"] = [
                    x for x in index["replays"]
                    if not (x["ticker"] == _active_ticker
                            and x["date"] == _active_date)
                ]
                _gh_write("replays/index.json", index,
                          f"Remove {_active_ticker} {_active_date} from index",
                          sha=index_sha)
            _gh_load_index.clear()
            del st.session_state["saved_replay_file"]
            del st.session_state["saved_replay_ticker"]
            del st.session_state["saved_replay_date"]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED MOVE
# ─────────────────────────────────────────────────────────────────────────────
def _render_expected_move(ticker: str, spot: float, df_gex, raw_df: pd.DataFrame, T: dict):
    """
    Expected Move tab — IV-derived price range projections.

    For each expiration we compute:
      EM_1σ = spot × atm_iv × √(dte/365)   [68% probability range]
      EM_2σ = EM_1σ × 2                     [95% probability range]

    We also show the straddle price (call_atm + put_atm) which is the market's
    direct dollar estimate of the expected move for that expiry.
    """

    # ── Gather per-expiry data ─────────────────────────────────────────────
    if raw_df.empty:
        st.warning("No options data available for Expected Move calculation.")
        return

    today = datetime.date.today()

    def _dte(exp_str):
        return max((datetime.datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days, 0)

    exps = sorted(raw_df["expiry"].unique(), key=_dte)

    exp_rows = []
    for exp in exps:
        d = raw_df[raw_df["expiry"] == exp].copy()
        dte_v = _dte(exp)
        if dte_v > 90:
            continue

        # ATM IV: use calls nearest to spot
        calls = d[d["flag"] == "C"].copy()
        calls["_dist"] = (calls["strike"] - spot).abs()
        atm_calls = calls.nsmallest(3, "_dist")
        atm_iv = float(atm_calls["iv"].mean()) if not atm_calls.empty else 0.0
        if atm_iv < 0.005:
            continue

        # 1σ and 2σ expected moves
        T_frac = max(dte_v, 0.5) / 365.0
        em_1s = spot * atm_iv * math.sqrt(T_frac)
        em_2s = em_1s * 2.0

        # Straddle price: ATM call + ATM put bid/ask mid
        atm_strike = float(atm_calls.nsmallest(1, "_dist")["strike"].iloc[0]) if not atm_calls.empty else spot
        strad_call = d[(d["flag"] == "C") & (d["strike"] == atm_strike)]
        strad_put  = d[(d["flag"] == "P") & (d["strike"] == atm_strike)]
        call_mid   = float(strad_call["last_price"].mean()) if not strad_call.empty else 0.0
        put_mid    = float(strad_put["last_price"].mean())  if not strad_put.empty  else 0.0
        straddle   = call_mid + put_mid

        # OI-weighted IV across all strikes for this expiry
        d_oi = d[d["open_interest"] > 0].copy()
        if not d_oi.empty and d_oi["open_interest"].sum() > 0:
            w_iv = float(np.average(d_oi["iv"], weights=d_oi["open_interest"]))
        else:
            w_iv = atm_iv

        exp_rows.append({
            "exp":        exp,
            "dte":        dte_v,
            "atm_iv":     atm_iv,
            "w_iv":       w_iv,
            "em_1s":      em_1s,
            "em_2s":      em_2s,
            "em_1s_pct":  em_1s / spot * 100,
            "em_2s_pct":  em_2s / spot * 100,
            "hi_1s":      spot + em_1s,
            "lo_1s":      spot - em_1s,
            "hi_2s":      spot + em_2s,
            "lo_2s":      spot - em_2s,
            "straddle":   straddle,
            "strad_pct":  straddle / spot * 100 if spot > 0 else 0.0,
        })

    if not exp_rows:
        st.warning("No valid ATM IV found for any expiration.")
        return

    exp_df = pd.DataFrame(exp_rows)

    # ── Header metrics strip ───────────────────────────────────────────────
    nearest = exp_df.iloc[0]
    n_dte   = int(nearest["dte"])
    n_iv    = nearest["atm_iv"] * 100
    n_em1   = nearest["em_1s"]
    n_em1p  = nearest["em_1s_pct"]
    n_strad = nearest["straddle"]

    def _card(label, value, color, sub=""):
        return f"""
        <div style="flex:1;min-width:110px;background:{T['bg1']};border:1px solid {T['line2']};
                    border-radius:6px;padding:11px 15px;border-top:2px solid {color};">
          <div style="font-family:'Barlow',sans-serif;font-size:8px;font-weight:600;
                      color:{T['t3']};letter-spacing:2px;text-transform:uppercase;
                      margin-bottom:6px;">{label}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:19px;font-weight:500;
                      color:{color};letter-spacing:-0.5px;">{value}</div>
          {f'<div style="font-size:8px;color:{T["t3"]};margin-top:3px;">{sub}</div>' if sub else ''}
        </div>"""

    cards = ""
    cards += _card("ATM IV",        f"{n_iv:.1f}%",           T["amber"],  f"nearest exp · {n_dte}DTE")
    cards += _card("1σ Move",       f"±${n_em1:.2f}",         T["green"],  f"±{n_em1p:.2f}%  68% prob")
    cards += _card("2σ Move",       f"±${nearest['em_2s']:.2f}", T["blue"], f"±{nearest['em_2s_pct']:.2f}%  95% prob")
    cards += _card("Straddle $",    f"${n_strad:.2f}",        T["violet"], f"{nearest['strad_pct']:.2f}% of spot")
    iv_rv  = compute_iv_rv_spread(raw_df, spot, ticker)
    ivrv_c = T["green"] if iv_rv > 0 else T["red"]
    cards += _card("IV − RV",       f"{iv_rv:+.1f}pp",        ivrv_c,      "ATM IV minus 20d HV")

    st.markdown(f'<div style="display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap;">{cards}</div>',
                unsafe_allow_html=True)

    # ── 1) Cone Chart — Expected Move over time ────────────────────────────
    st.markdown(f'<div class="sub-head">Expected Move Cone</div>', unsafe_allow_html=True)

    # Use the first expiry's ATM IV as the "constant" IV for the cone
    # so the cone is smooth and not jagged from per-expiry IV changes.
    base_iv  = float(exp_df["atm_iv"].iloc[0])
    # Build continuous cone from 0 → max DTE across all exps
    max_dte  = int(exp_df["dte"].max())
    cone_dte = np.arange(0, max_dte + 2, 1)
    cone_1s  = spot * base_iv * np.sqrt(cone_dte / 365.0)
    cone_2s  = cone_1s * 2.0

    fig_cone = go.Figure()

    # 2σ band (outer, faint)
    fig_cone.add_trace(go.Scatter(
        x=cone_dte, y=spot + cone_2s,
        mode="lines", line=dict(color=T["blue"], width=1, dash="dot"),
        name="+2σ", hovertemplate="DTE %{x}<br>+2σ: $%{y:.2f}<extra></extra>",
    ))
    fig_cone.add_trace(go.Scatter(
        x=cone_dte, y=spot - cone_2s,
        mode="lines", line=dict(color=T["blue"], width=1, dash="dot"),
        fill="tonexty",
        fillcolor=f"rgba({int(T['blue'][1:3],16)},{int(T['blue'][3:5],16)},{int(T['blue'][5:7],16)},0.06)",
        name="-2σ", hovertemplate="DTE %{x}<br>-2σ: $%{y:.2f}<extra></extra>",
    ))

    # 1σ band (inner)
    fig_cone.add_trace(go.Scatter(
        x=cone_dte, y=spot + cone_1s,
        mode="lines", line=dict(color=T["green"], width=1.5, dash="dash"),
        name="+1σ", hovertemplate="DTE %{x}<br>+1σ: $%{y:.2f}<extra></extra>",
    ))
    fig_cone.add_trace(go.Scatter(
        x=cone_dte, y=spot - cone_1s,
        mode="lines", line=dict(color=T["red"], width=1.5, dash="dash"),
        fill="tonexty",
        fillcolor=f"rgba({int(T['green'][1:3],16)},{int(T['green'][3:5],16)},{int(T['green'][5:7],16)},0.08)",
        name="-1σ", hovertemplate="DTE %{x}<br>-1σ: $%{y:.2f}<extra></extra>",
    ))

    # Spot line
    fig_cone.add_hline(y=spot, line_color=T["t1"], line_width=1.5,
                       annotation_text=f"  Spot ${spot:.2f}",
                       annotation_font_color=T["t1"], annotation_font_size=9)

    # Expiry marker pins
    for _, row in exp_df.iterrows():
        _x = int(row["dte"])
        _hi = row["hi_1s"]
        _lo = row["lo_1s"]
        fig_cone.add_shape(type="line", x0=_x, x1=_x, y0=_lo, y1=_hi,
                           line=dict(color=T["amber"], width=1, dash="dot"))
        fig_cone.add_trace(go.Scatter(
            x=[_x], y=[_hi],
            mode="markers+text",
            marker=dict(color=T["amber"], size=7, line=dict(color=T["bg"], width=1.5)),
            text=[row["exp"][5:]],
            textposition="top center",
            textfont=dict(size=8, color=T["amber"], family="JetBrains Mono"),
            hovertemplate=f"<b>{row['exp']}</b>  {_x}DTE<br>"
                          f"ATM IV: {row['atm_iv']*100:.1f}%<br>"
                          f"1σ ±${row['em_1s']:.2f} (±{row['em_1s_pct']:.2f}%)<br>"
                          f"2σ ±${row['em_2s']:.2f} (±{row['em_2s_pct']:.2f}%)<br>"
                          f"Straddle: ${row['straddle']:.2f}<extra></extra>",
            showlegend=False,
        ))
        fig_cone.add_trace(go.Scatter(
            x=[_x], y=[_lo],
            mode="markers",
            marker=dict(color=T["red"], size=7, line=dict(color=T["bg"], width=1.5)),
            hoverinfo="skip", showlegend=False,
        ))

    _cone_layout = {**PLOTLY_BASE, "showlegend": True}
    fig_cone.update_layout(
        **_cone_layout,
        height=340,
        legend=dict(orientation="h", y=1.08, x=0,
                    font=dict(size=9, color=TEXT2, family="JetBrains Mono"),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, r=90, b=50, l=60),
        xaxis=dict(
            title=dict(text="Days to Expiration", font=dict(size=9, color=TEXT3)),
            gridcolor=LINE, gridwidth=1, zerolinecolor=LINE2,
            tickfont=dict(size=9, family="JetBrains Mono"),
        ),
        yaxis=dict(
            title=dict(text="Price", font=dict(size=9, color=TEXT3)),
            gridcolor=LINE, gridwidth=1, tickprefix="$",
            tickfont=dict(size=9, family="JetBrains Mono"),
        ),
        hoverlabel=dict(bgcolor=T["bg2"], bordercolor=T["line_bright"],
                        font=dict(family="JetBrains Mono", size=10, color=T["t1"])),
    )
    st.plotly_chart(fig_cone, use_container_width=True, config={"displayModeBar": False})

    # ── 2) Per-Expiry Price Ladder ─────────────────────────────────────────
    st.markdown(f'<div class="sub-head">Per-Expiry Expected Range</div>', unsafe_allow_html=True)

    fig_lad = go.Figure()
    exp_labels = [f"{row['exp'][5:]}  {int(row['dte'])}d" for _, row in exp_df.iterrows()]
    exp_idx    = list(range(len(exp_df)))

    # 2σ bars (background, faint)
    for i, (_, row) in enumerate(exp_df.iterrows()):
        fig_lad.add_shape(type="rect",
            x0=row["lo_2s"], x1=row["hi_2s"], y0=i - 0.35, y1=i + 0.35,
            fillcolor=f"rgba({int(T['blue'][1:3],16)},{int(T['blue'][3:5],16)},{int(T['blue'][5:7],16)},0.10)",
            line=dict(color=T["blue"], width=1, dash="dot"),
        )
        # 1σ bar (inner)
        fig_lad.add_shape(type="rect",
            x0=row["lo_1s"], x1=row["hi_1s"], y0=i - 0.22, y1=i + 0.22,
            fillcolor=f"rgba({int(T['green'][1:3],16)},{int(T['green'][3:5],16)},{int(T['green'][5:7],16)},0.14)",
            line=dict(color=T["green"], width=1.2),
        )
        # Straddle price markers
        if row["straddle"] > 0:
            fig_lad.add_shape(type="line",
                x0=spot - row["straddle"], x1=spot - row["straddle"],
                y0=i - 0.35, y1=i + 0.35,
                line=dict(color=T["violet"], width=1.5, dash="dot"),
            )
            fig_lad.add_shape(type="line",
                x0=spot + row["straddle"], x1=spot + row["straddle"],
                y0=i - 0.35, y1=i + 0.35,
                line=dict(color=T["violet"], width=1.5, dash="dot"),
            )

    # Invisible hover traces
    for i, (_, row) in enumerate(exp_df.iterrows()):
        strad_note = f"${row['straddle']:.2f} straddle ({row['strad_pct']:.2f}%)" if row["straddle"] > 0 else "no straddle data"
        for px, lbl in [(row["hi_1s"], "+1σ"), (row["lo_1s"], "-1σ"),
                        (row["hi_2s"], "+2σ"), (row["lo_2s"], "-2σ")]:
            fig_lad.add_trace(go.Scatter(
                x=[px], y=[i],
                mode="markers",
                marker=dict(
                    color=T["green"] if "+" in lbl else T["red"],
                    size=9, symbol="line-ns",
                    line=dict(width=2, color=T["green"] if "+" in lbl else T["red"]),
                ),
                hovertemplate=(
                    f"<b>{row['exp']}  {int(row['dte'])}DTE</b><br>"
                    f"ATM IV: {row['atm_iv']*100:.1f}%<br>"
                    f"1σ: ${row['lo_1s']:.2f} ↔ ${row['hi_1s']:.2f}  (±{row['em_1s_pct']:.2f}%)<br>"
                    f"2σ: ${row['lo_2s']:.2f} ↔ ${row['hi_2s']:.2f}  (±{row['em_2s_pct']:.2f}%)<br>"
                    f"Straddle: {strad_note}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

    # Spot vertical line
    fig_lad.add_vline(x=spot, line_color=T["t1"], line_width=2,
                      annotation_text=f"  Spot ${spot:.2f}",
                      annotation_font_color=T["t1"], annotation_font_size=9)

    all_prices = []
    for _, row in exp_df.iterrows():
        all_prices += [row["lo_2s"], row["hi_2s"]]
    x_pad = (max(all_prices) - min(all_prices)) * 0.08

    fig_lad.update_layout(
        **PLOTLY_BASE,
        height=max(260, len(exp_df) * 70 + 80),
        margin=dict(t=10, r=90, b=50, l=60),
        xaxis=dict(
            title=dict(text="Price", font=dict(size=9, color=TEXT3)),
            range=[min(all_prices) - x_pad, max(all_prices) + x_pad],
            gridcolor=LINE, gridwidth=1, zerolinecolor=LINE2, tickprefix="$",
            tickfont=dict(size=9, family="JetBrains Mono"),
        ),
        yaxis=dict(
            tickvals=exp_idx, ticktext=exp_labels,
            tickfont=dict(size=9, family="JetBrains Mono"),
            gridcolor=LINE, gridwidth=1,
        ),
        hoverlabel=dict(bgcolor=T["bg2"], bordercolor=T["line_bright"],
                        font=dict(family="JetBrains Mono", size=10, color=T["t1"])),
    )
    st.plotly_chart(fig_lad, use_container_width=True, config={"displayModeBar": False})

    # ── 3) IV Term Structure ───────────────────────────────────────────────
    st.markdown(f'<div class="sub-head">IV Term Structure</div>', unsafe_allow_html=True)

    fig_ts = go.Figure()
    ts_colors = [T["green"] if iv <= float(exp_df["atm_iv"].iloc[0]) * 1.05
                 else T["amber"]
                 for iv in exp_df["atm_iv"]]

    fig_ts.add_trace(go.Scatter(
        x=exp_df["dte"].tolist(),
        y=(exp_df["atm_iv"] * 100).tolist(),
        mode="lines+markers",
        line=dict(color=T["t3"], width=1.2),
        marker=dict(color=ts_colors, size=9, line=dict(color=T["bg"], width=1.5)),
        name="ATM IV",
        customdata=np.stack([exp_df["exp"], exp_df["em_1s_pct"]], axis=1),
        hovertemplate=(
            "<b>%{customdata[0]}</b>  %{x}DTE<br>"
            "ATM IV: <b>%{y:.2f}%</b><br>"
            "1σ move: ±%{customdata[1]:.2f}%"
            "<extra></extra>"
        ),
    ))
    fig_ts.add_trace(go.Scatter(
        x=exp_df["dte"].tolist(),
        y=(exp_df["w_iv"] * 100).tolist(),
        mode="lines",
        line=dict(color=T["violet"], width=1, dash="dot"),
        name="OI-Wtd IV",
        hovertemplate="OI-weighted IV: %{y:.2f}%<extra></extra>",
    ))

    # Horizontal reference: current ATM IV
    fig_ts.add_hline(y=float(exp_df["atm_iv"].iloc[0]) * 100,
                     line_dash="dot", line_color=T["t3"], line_width=1,
                     annotation_text=f"  Near ATM IV {float(exp_df['atm_iv'].iloc[0])*100:.1f}%",
                     annotation_font_color=T["t3"], annotation_font_size=9)

    _ts_layout = {**PLOTLY_BASE, "showlegend": True}
    fig_ts.update_layout(
        **_ts_layout,
        height=240,
        legend=dict(orientation="h", y=1.12, x=0,
                    font=dict(size=9, color=TEXT2, family="JetBrains Mono"),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, r=90, b=50, l=60),
        xaxis=dict(
            title=dict(text="Days to Expiration", font=dict(size=9, color=TEXT3)),
            gridcolor=LINE, gridwidth=1,
            tickfont=dict(size=9, family="JetBrains Mono"),
        ),
        yaxis=dict(
            title=dict(text="IV (%)", font=dict(size=9, color=TEXT3)),
            gridcolor=LINE, gridwidth=1,
            tickfont=dict(size=9, family="JetBrains Mono"),
            ticksuffix="%",
        ),
        hoverlabel=dict(bgcolor=T["bg2"], bordercolor=T["line_bright"],
                        font=dict(family="JetBrains Mono", size=10, color=T["t1"])),
    )
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

    # ── 4) Summary Table ───────────────────────────────────────────────────
    st.markdown(f'<div class="sub-head">Expected Move Summary</div>', unsafe_allow_html=True)
    tbl = exp_df[["exp","dte","atm_iv","em_1s","em_1s_pct","em_2s","em_2s_pct","straddle","strad_pct"]].copy()
    tbl.columns = ["Expiry","DTE","ATM IV","1σ Move $","1σ Move %","2σ Move $","2σ Move %","Straddle $","Straddle %"]

    def _em_style(val):
        _ts = get_theme()
        return f"color:{_ts['green']};"

    st.dataframe(
        tbl.style
           .format({
               "ATM IV":     "{:.2%}",
               "1σ Move $":  "±${:.2f}",
               "1σ Move %":  "±{:.2f}%",
               "2σ Move $":  "±${:.2f}",
               "2σ Move %":  "±{:.2f}%",
               "Straddle $": "${:.2f}",
               "Straddle %": "{:.2f}%",
           })
           .map(_em_style, subset=["1σ Move $","2σ Move $"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(f"""
    <div style="margin-top:12px;padding:10px 14px;background:{T['bg1']};
                border:1px solid {T['line']};border-radius:4px;
                border-left:3px solid {T['amber']};">
      <div style="font-family:'Barlow',sans-serif;font-size:9px;font-weight:600;
                  color:{T['amber']};letter-spacing:1.5px;text-transform:uppercase;
                  margin-bottom:3px;">Methodology</div>
      <div style="font-family:'Barlow',sans-serif;font-size:9.5px;color:{T['t2']};line-height:1.7;">
        <b>1σ Expected Move</b> = Spot × ATM IV × √(DTE/365) &nbsp;·&nbsp; 68.3% probability range<br>
        <b>2σ Expected Move</b> = 1σ × 2 &nbsp;·&nbsp; 95.5% probability range<br>
        <b>Straddle price</b> = market's direct dollar estimate (ATM call mid + ATM put mid)<br>
        <b>OI-Weighted IV</b> = IV averaged across all strikes, weighted by open interest<br>
        IV source: CBOE delayed feed (~15 min). OI is end-of-day. Not financial advice.
      </div>
    </div>
    """, unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD FRAGMENT
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def dashboard():

    T            = get_theme()
    asset_toggle = st.session_state.asset_choice
    max_exp      = st.session_state.max_exp

    _equiv_map = {
        "SPY": ("ES Equiv", 10), "QQQ": ("NQ Equiv", 47.5),
        "SPX": ("ES Equiv", 1),  "NDX": ("NQ Equiv", 1),
        "RUT": ("RTY Equiv", 5), "IWM": ("RTY Equiv", 5),
    }
    equiv_label_fb, _ = _equiv_map.get(asset_toggle, (f"{asset_toggle} $Val", 1))

    try:
        _spot_fr = get_spot(asset_toggle)
    except Exception:
        _spot_fr = 500.0
    equiv_label, equiv_mult = _get_equiv_config(asset_toggle, _spot_fr)

    try:
        result = fetch_options_data(asset_toggle, max_exp)
    except Exception as _e:
        T2 = get_theme()
        st.markdown(f"""
        <div style="background:{T2['bg1']};border:1px solid {T2['red']};border-left:3px solid {T2['red']};
                    border-radius:4px;padding:16px 20px;margin:8px 0;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;
                      letter-spacing:2px;color:{T2['red']};text-transform:uppercase;margin-bottom:8px;">
            CBOE Fetch Failed
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{T2['t3']};
                      margin-bottom:8px;word-break:break-all;">
            {type(_e).__name__}: {str(_e)[:300]}
          </div>
          <div style="font-family:'Barlow',sans-serif;font-size:11px;color:{T2['t2']};line-height:1.8;">
            <b>Common causes:</b><br>
            &nbsp;· CBOE CDN temporarily unreachable (HTTP 4xx/5xx)<br>
            &nbsp;· Network timeout — cloud deployment may be blocked<br>
            &nbsp;· CBOE changed their API URL structure<br>
            Try clearing cache and retrying. If it persists after market open, the CDN URL may have changed.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Clear Cache + Retry", key="retry_fetch_btn"):
            for k in list(st.session_state.keys()):
                if k.startswith("_chain_") or k.startswith("_spot_"):
                    del st.session_state[k]
            st.rerun()
        return

    df, _cboe_spot, raw_df = result if len(result) == 3 else (pd.DataFrame(), 580.0, pd.DataFrame())
    # Override CBOE delayed spot with real-time Yahoo price.
    # _spot_fr was already fetched from Yahoo above; reuse it.
    # If Yahoo failed, _spot_fr == 500.0 fallback — in that case keep CBOE.
    spot_price = _spot_fr if _spot_fr != 500.0 else _cboe_spot
    if df.empty:
        T2 = get_theme()
        st.markdown(f"""
        <div style="background:{T2['bg1']};border:1px solid {T2['red']};border-left:3px solid {T2['red']};
                    border-radius:4px;padding:16px 20px;margin:8px 0;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;
                      letter-spacing:2px;color:{T2['red']};text-transform:uppercase;margin-bottom:8px;">
            No Qualifying Options Data
          </div>
          <div style="font-family:'Barlow',sans-serif;font-size:11px;color:{T2['t2']};line-height:1.8;">
            CBOE returned a response but zero options passed the data quality filters.<br>
            <b>Most common causes:</b><br>
            &nbsp;· Market is closed — bids/asks are zero so no IV can be solved<br>
            &nbsp;· CBOE delayed feed went stale (try clearing cache below)<br>
            &nbsp;· All near-term strikes have OI &lt; 100 (low-liquidity asset)<br>
            &nbsp;· Temporary CBOE CDN issue — wait 30–60 seconds and retry
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Clear Cache + Retry", key="retry_clear_btn"):
            for k in list(st.session_state.keys()):
                if k.startswith("_chain_") or k.startswith("_spot_"):
                    del st.session_state[k]
            st.rerun()
        return

    df["es_strike"] = df["strike"] * equiv_mult

    gamma_flip, call_wall, put_wall, max_pain = compute_key_levels(df, spot_price, raw_df)
    vol_trigger, mom_wall, mom_val = compute_intraday_levels(df, spot_price)

    is_long_gamma     = spot_price > gamma_flip
    regime_color      = T["green"] if is_long_gamma else T["red"]
    regime_label      = "LONG GAMMA  ·  STABLE" if is_long_gamma else "SHORT GAMMA  ·  VOLATILE"

    total_net_gex     = df["gex_net"].sum()
    total_net_vol_gex = df["vol_gex_net"].sum()
    total_call_gex    = df["call_gex"].sum()
    total_put_gex     = df["put_gex"].sum()
    gex_ratio = total_call_gex / (total_call_gex + abs(total_put_gex)) \
                if (total_call_gex + abs(total_put_gex)) > 0 else 0.5

    if is_long_gamma:
        bias_note, bias_color = "BUY DIPS",  T["green"]
    else:
        bias_note, bias_color = "SELL RIPS", T["red"]

    total_dex = df["dex_net"].sum() if "dex_net" in df.columns else 0.0
    total_vex = df["vex_net"].sum() if "vex_net" in df.columns else 0.0
    total_cex = df["cex_net"].sum() if "cex_net" in df.columns else 0.0

    _atm_mask_df = (df["dist_pct"].abs() <= 0.5) if "dist_pct" in df.columns else pd.Series([True]*len(df))
    _atm_iv_df   = df.loc[_atm_mask_df, "iv"] if "iv" in df.columns else pd.Series(dtype=float)
    atm_iv_pct   = float(_atm_iv_df.mean() * 100) if not _atm_iv_df.empty else 0.0

    iv_rv_spread = compute_iv_rv_spread(raw_df, spot_price, asset_toggle)
    flow_ratio, net_flow = compute_flow(raw_df, spot_price)
    flow_color = T["green"] if flow_ratio >= 0.5 else T["red"]

    mom_color = T["blue"] if mom_val >= 0 else T["violet"]
    mom_label = "Momentum Wall · Call" if mom_val >= 0 else "Momentum Wall · Put"

    if abs(net_flow) >= 1e6:
        nf_str = f"{net_flow/1e6:+.2f}M"
    elif abs(net_flow) >= 1e3:
        nf_str = f"{net_flow/1e3:+.1f}K"
    else:
        nf_str = f"{net_flow:+.0f}"

    st.session_state["_sb_metrics"] = dict(
        spot_price    = spot_price,
        gamma_flip    = gamma_flip,
        call_wall     = call_wall,
        put_wall      = put_wall,
        max_pain      = max_pain,
        total_net_gex = total_net_gex,
        total_net_vol_gex = total_net_vol_gex,
        gex_ratio     = gex_ratio,
        vol_trigger   = vol_trigger,
        mom_wall      = mom_wall,
        mom_val       = mom_val,
        iv_rv_spread  = iv_rv_spread,
        flow_ratio    = flow_ratio,
        net_flow      = net_flow,
        nf_str        = nf_str,
    )

    # ── Regime Banner ──────────────────────────────────────────────────────
    bar_grad = f"linear-gradient(90deg, {regime_color} 0%, transparent 60%)"
    glow_col = T["green_glow"] if is_long_gamma else T["red_glow"]

    # Detect whether spot came from live Yahoo or delayed CBOE
    _spot_cache = st.session_state.get(f"_spot_{asset_toggle}", {})
    _spot_src   = _spot_cache.get("src", "delayed")
    _spot_live  = _spot_src == "live"
    _spot_badge = (
        f"<span style='font-size:8px;font-weight:700;letter-spacing:1.5px;"
        f"color:{T['green']};border:1px solid {T['green']}33;"
        f"border-radius:3px;padding:1px 5px;margin-left:4px;'>LIVE</span>"
        if _spot_live else
        f"<span style='font-size:8px;font-weight:600;letter-spacing:1.5px;"
        f"color:{T['amber']};border:1px solid {T['amber']}33;"
        f"border-radius:3px;padding:1px 5px;margin-left:4px;'>~15m DELAY</span>"
    )

    st.markdown(f"""
    <div class="regime" style="--bar-grad:{bar_grad}; --bar-glow:radial-gradient(ellipse at 0% 50%, {glow_col}, transparent 65%);">
      <div>
        <div class="regime-meta">
          {asset_toggle} &nbsp;·&nbsp; ${spot_price:.2f} {_spot_badge}
          &nbsp;·&nbsp; OI-GEX {total_net_gex:+.3f}B
          &nbsp;·&nbsp; Vol-GEX {total_net_vol_gex:+.3f}B
          &nbsp;·&nbsp; {max_exp} exp
          &nbsp;·&nbsp; <span style='color:{T["t4"]};font-size:8px;'>chain: ~15m delay · OI: EOD</span>
        </div>
        <div class="regime-state" style="color:{regime_color}">{regime_label}</div>
      </div>
      <div class="regime-right">
        <div class="regime-bias-label">Structural Bias</div>
        <div class="regime-bias-value" style="color:{bias_color}">{bias_note}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode Buttons + Refresh Timer ───────────────────────────────────────
        _modes_list  = ["GEX", "HEAT", "MOVE", "DAILY", "REPLAY", "SAVES"]
        _labels_list = ["OI GEX", "Heatmap", "Exp. Move", "Daily Levels", "⏱ Replay", "📼 Saved"]
        _btn_cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        for _col, _mode, _lbl in zip(_btn_cols[:6], _modes_list, _labels_list):
            with _col:
                if st.button(_lbl, key=f"mode_btn_{_mode}",
                             type="primary" if st.session_state.radar_mode == _mode else "secondary"):
                    st.session_state.radar_mode = _mode
                    if _mode == "REPLAY":
                        st.session_state.replay_playing = False
                    st.rerun()

        with _btn_cols[6]:
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; height:38px;">
              <div style="
                display:inline-flex; align-items:center; gap:7px;
                background:{T['bg1']}; border:1px solid {T['line2']};
                border-radius:20px; padding:5px 13px;
                font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:500;
                letter-spacing:1px; color:var(--text-2); white-space:nowrap;">
                <div style="width:5px;height:5px;border-radius:50%;background:var(--amber);flex-shrink:0;
                            animation:tpblink 1s ease-in-out infinite;"></div>
                REFRESH <span id="gex-cdown" style="color:var(--amber);font-weight:600;">{AUTO_REFRESH_SECONDS}</span>s
              </div>
            </div>
            <style>@keyframes tpblink{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}</style>
            """, unsafe_allow_html=True)

        import time as _time
        st.markdown(
            f'<div id="gex-refresh-signal" style="display:none">{_time.time()}</div>',
            unsafe_allow_html=True
        )

        if st.session_state.radar_mode not in ("GEX", "HEAT", "MOVE", "DAILY", "REPLAY", "SAVES", "BACKTEST"):
            st.session_state.radar_mode = "GEX"

    # ── DEX / VEX / CEX / IV Exposure Strip ────────────────────────────────
    _dex_col  = T["green"] if total_dex >= 0 else T["red"]
    _vex_col  = T["green"] if total_vex >= 0 else T["red"]
    _cex_col  = T["blue"] if total_cex >= 0 else T["violet"]
    _iv_col   = T["amber"]

    def _fmt_exp(v, unit="M"):
        if abs(v) >= 1000:
            return f"{v/1000:+.2f}B"
        return f"{v:+.2f}{unit}"

    st.markdown(f"""
    <div style="display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap;">
      <div style="flex:1; min-width:110px; background:{T['bg1']}; border:1px solid {T['line2']};
                  border-radius:6px; padding:10px 14px;">
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; font-weight:600;
                    color:{T['t3']}; letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:5px;">ATM IV</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:17px; font-weight:500;
                    color:{_iv_col}; letter-spacing:-0.5px;">{atm_iv_pct:.1f}%</div>
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; color:{T['t3']};
                    margin-top:3px; letter-spacing:0.5px;">Impl. vol nearest exp</div>
      </div>
      <div style="flex:1; min-width:110px; background:{T['bg1']}; border:1px solid {T['line2']};
                  border-radius:6px; padding:10px 14px;">
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; font-weight:600;
                    color:{T['t3']}; letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:5px;">DEX</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:17px; font-weight:500;
                    color:{_dex_col}; letter-spacing:-0.5px;">{_fmt_exp(total_dex/1e6)}</div>
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; color:{T['t3']};
                    margin-top:3px; letter-spacing:0.5px;">Delta exposure $</div>
      </div>
      <div style="flex:1; min-width:110px; background:{T['bg1']}; border:1px solid {T['line2']};
                  border-radius:6px; padding:10px 14px; position:relative;">
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; font-weight:600;
                    color:{T['t3']}; letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:5px;">VEX</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:17px; font-weight:500;
                    color:{_vex_col}; letter-spacing:-0.5px;">{_fmt_exp(total_vex)}</div>
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; color:{T['t3']};
                    margin-top:3px; letter-spacing:0.5px;">Vega exp · EOD signal</div>
      </div>
      <div style="flex:1; min-width:110px; background:{T['bg1']}; border:1px solid {T['line2']};
                  border-radius:6px; padding:10px 14px;">
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; font-weight:600;
                    color:{T['t3']}; letter-spacing:2px; text-transform:uppercase;
                    margin-bottom:5px;">CEX</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:17px; font-weight:500;
                    color:{_cex_col}; letter-spacing:-0.5px;">{_fmt_exp(total_cex)}</div>
        <div style="font-family:'Barlow, sans-serif',sans-serif; font-size:8px; color:{T['t3']};
                    margin-top:3px; letter-spacing:0.5px;">Charm exp · news/events</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    plot_df = df.copy()
    spacing = float(plot_df["strike"].diff().median()) if len(plot_df) > 1 else 1.0

    # ── OI GEX chart ──────────────────────────────────────────────────────
    if st.session_state.radar_mode == "GEX":
        chart_col, info_col = st.columns([9, 3])

        with chart_col:
            colors = [T["bar_pos"] if v >= 0 else T["bar_neg"] for v in plot_df["gex_net"]]
            fig = go.Figure(gex_bars(
                y=plot_df["strike"], x=plot_df["gex_net"], spacing=spacing,
                colors=colors,
                customdata=np.stack([plot_df["es_strike"], plot_df["call_gex"],
                                     plot_df["put_gex"], plot_df["open_interest"],
                                     plot_df["vol_gex_net"]], axis=1),
                hovertemplate=(
                    f"<b>Strike %{{y}}</b><br>{equiv_label}: %{{customdata[0]:.2f}}<br>"
                    "Net OI-GEX: <b>%{x:.4f}B</b><br>"
                    "  Call: +%{customdata[1]:.4f}B<br>"
                    "  Put:  %{customdata[2]:.4f}B<br>"
                    "Net Vol-GEX: %{customdata[4]:.4f}B<br>"
                    "Open Interest: %{customdata[3]:,.0f}<extra></extra>"
                )
            ))
            add_reference_lines(fig, spot_price, gamma_flip)
            bar_layout(fig, plot_df["gex_net"], "Net OI-GEX ($B)", spot_price)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with info_col:
            st.markdown(
                render_kl_panel(spot_price, gamma_flip, call_wall, put_wall,
                                max_pain, vol_trigger, mom_wall, mom_val),
                unsafe_allow_html=True
            )
            st.markdown('<div class="sub-head" style="margin-top:14px;">GEX Topography</div>', unsafe_allow_html=True)
            _landscape_fig = build_gex_landscape(df, spot_price)
            if _landscape_fig is not None:
                st.plotly_chart(_landscape_fig, use_container_width=True, config={"displayModeBar": False})

    # ── HEATMAP ───────────────────────────────────────────────────────────
    elif st.session_state.radar_mode == "HEAT":

        if "heat_dual" not in st.session_state:
            st.session_state.heat_dual = False

        _label = f"{'☑' if st.session_state.heat_dual else '☐'}  SPY + QQQ DUAL"
        st.markdown('<div class="dual-btn-wrap">', unsafe_allow_html=True)
        if st.button(_label, key="heat_dual_btn"):
            st.session_state.heat_dual = not st.session_state.heat_dual
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── NEW premium heatmap function ──────────────────────────────────
        def make_heatmap_fig(raw, spot, lbl, gflip, c_wall=None, p_wall=None):
            strikes, expiries, z = build_heatmap_matrix(raw, spot, mode="oi")
            if strikes is None:
                f = go.Figure()
                f.add_annotation(
                    text="No heatmap data available",
                    showarrow=False,
                    font=dict(color=TEXT3, size=13, family="JetBrains Mono"),
                )
                f.update_layout(**PLOTLY_BASE, height=500)
                return f

            exp_labels  = [e if isinstance(e, str) else str(e) for e in expiries]
            strike_nums = [float(s) for s in strikes]
            z_arr       = np.array(z, dtype=float)

            # ── Resolve call/put walls from raw data if not passed in ─────
            def _walls_from_raw(raw_df_inner, spot_inner):
                if raw_df_inner is None or raw_df_inner.empty:
                    return spot_inner * 1.01, spot_inner * 0.99
                try:
                    _ag = raw_df_inner.groupby("strike").agg(
                        call_gex=("call_gex", "sum"),
                        put_gex=("put_gex", "sum"),
                    ).reset_index()
                    _ag["gex_net"] = _ag["call_gex"] + _ag["put_gex"]
                    _p = _ag[_ag["gex_net"] > 0]
                    _n = _ag[_ag["gex_net"] < 0]
                    cw = float(_p.loc[_p["gex_net"].idxmax(), "strike"]) if not _p.empty else spot_inner * 1.01
                    pw = float(_n.loc[_n["gex_net"].idxmin(), "strike"]) if not _n.empty else spot_inner * 0.99
                    return cw, pw
                except Exception:
                    return spot_inner * 1.01, spot_inner * 0.99

            if c_wall is None or p_wall is None:
                _cw, _pw = _walls_from_raw(raw, spot)
                c_wall = c_wall if c_wall is not None else _cw
                p_wall = p_wall if p_wall is not None else _pw

            # ── Per-column √-normalisation ────────────────────────────────
            # Each expiry column is scaled to its own maximum so near-term
            # and far-dated strikes are visually comparable. The √0.45 power
            # compression makes walls explode bright while subtle mid-range
            # zones remain visible rather than being lost to the dominant expiry.
            col_max = np.abs(z_arr).max(axis=0, keepdims=True)
            col_max = np.where(col_max < 1e-12, 1e-12, col_max)
            z_norm  = z_arr / col_max                                       # [-1, 1] per column
            z_disp  = np.sign(z_norm) * np.power(np.abs(z_norm), 0.45)     # √ compression

            # Net strike profile — sum across all expiries for side bar
            net_per_strike = z_arr.sum(axis=1)

            # ── Thermal diverging colorscale ──────────────────────────────
            # Black = zero GEX (neutral)
            # Negative (puts):  deep red → vivid red → hot amber at max
            # Positive (calls): deep green → vivid green → blue-white at max
            # Darker = lower intensity, Brighter/hotter = more GEX — exactly
            # like a thermal camera: you can't miss the hot zones.
            _bg  = T["bg"]
            _red = T["red"]
            _grn = T["green"]
            _amb = T["amber"]
            _bl  = T["blue"]
            heat_cs = [
                [0.000, _amb],           # strongest puts  → amber "overload"
                [0.040, _red],
                [0.120, "#CC1800"],
                [0.250, "#770D00"],
                [0.390, "#380500"],
                [0.460, "#1A0200"],
                [0.490, "#0D0100"],
                [0.500, _bg],            # zero GEX → pure background colour
                [0.510, "#000D06"],
                [0.540, "#001A0B"],
                [0.610, "#003D18"],
                [0.750, "#007A30"],
                [0.880, _grn],
                [0.960, _bl],            # strongest calls → blue-white overload
                [1.000, "#DFFFFF"],
            ]

            # ── Subplot: heatmap (left 82%) + strike profile bar (right 18%) ─
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.82, 0.18],
                shared_yaxes=True,
                horizontal_spacing=0.015,
            )

            # Heatmap trace
            fig.add_trace(go.Heatmap(
                z=z_disp,
                x=exp_labels,
                y=strike_nums,
                customdata=z_arr,
                colorscale=heat_cs,
                zmid=0, zmin=-1.0, zmax=1.0,
                colorbar=dict(
                    title=dict(
                        text="GEX Intensity",
                        font=dict(size=8, color=TEXT3, family="JetBrains Mono"),
                        side="right",
                    ),
                    tickfont=dict(size=7.5, color=TEXT3, family="JetBrains Mono"),
                    tickvals=[-1.0, -0.65, 0.0, 0.65, 1.0],
                    ticktext=["MAX PUT", "MID PUT", "NEUTRAL", "MID CALL", "MAX CALL"],
                    thickness=10,
                    len=0.70,
                    x=0.825,
                    xanchor="right",
                    tickcolor=TEXT3,
                    outlinewidth=0,
                    bgcolor=T["bg1"],
                    bordercolor=T["line2"],
                    borderwidth=0,
                ),
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    "Strike: <b>$%{y:.2f}</b><br>"
                    "Expiry: %{x}<br>"
                    "Net OI-GEX: <b>%{customdata:.4f}B</b>"
                    "<extra></extra>"
                ),
                xgap=2.5,
                ygap=1.5,
                zsmooth=False,
            ), row=1, col=1)

            # Strike profile bar (right panel) — net GEX across all expiries per strike
            _sp_colors = [T["bar_pos"] if v >= 0 else T["bar_neg"] for v in net_per_strike]
            _sp_spacing = (
                [(strike_nums[i + 1] - strike_nums[i]) * 0.78 if i + 1 < len(strike_nums)
                 else (strike_nums[-1] - strike_nums[-2]) * 0.78
                 for i in range(len(strike_nums))]
                if len(strike_nums) > 1 else [1.0]
            )
            fig.add_trace(go.Bar(
                x=net_per_strike,
                y=strike_nums,
                orientation="h",
                marker=dict(color=_sp_colors, line=dict(width=0), opacity=0.88),
                width=_sp_spacing,
                hovertemplate=(
                    "Strike $%{y:.2f}<br>"
                    "Net GEX (all exp): <b>%{x:.4f}B</b>"
                    "<extra></extra>"
                ),
            ), row=1, col=2)

            # ── Key level lines across full chart width ───────────────────
            _level_defs = [
                (spot,    T["t1"],    2.5, "solid",  f"SPOT  ${spot:.2f}"),
                (gflip,   _amb,       1.3, "dot",    f"ZERO Γ  ${gflip:.2f}"),
                (c_wall,  _grn,       1.3, "dash",   f"CALL WALL  ${c_wall:.2f}"),
                (p_wall,  _red,       1.3, "dash",   f"PUT WALL  ${p_wall:.2f}"),
            ]
            s_lo = strike_nums[0]
            s_hi = strike_nums[-1]

            for _price, _col, _w, _dash, _ann_lbl in _level_defs:
                # Only draw if the level falls within the visible strike range (±0.5% tolerance)
                if not (s_lo * 0.995 <= _price <= s_hi * 1.005):
                    continue
                fig.add_shape(
                    type="line", x0=0, x1=1, xref="paper",
                    y0=_price, y1=_price, yref="y",
                    line=dict(color=_col, width=_w, dash=_dash),
                    layer="above",
                )
                # Pill-style annotation flush to right edge
                fig.add_annotation(
                    x=1.01, y=_price, xref="paper", yref="y",
                    text=f" {_ann_lbl} ",
                    showarrow=False, xanchor="left",
                    font=dict(size=8, color=_col, family="JetBrains Mono", weight=600),
                    bgcolor=T["bg1"],
                    bordercolor=_col,
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.95,
                )

            # ── Layout ────────────────────────────────────────────────────
            fig.update_layout(
                **PLOTLY_BASE,
                barmode="relative",
                bargap=0.06,
                title=dict(
                    text=(
                        f"<b>{lbl}</b>  ·  GEX HEATMAP  "
                        f"<span style='font-size:9px;color:{TEXT3};'>"
                        f"per-expiry √-normalised intensity  ·  "
                        f"brighter = higher GEX  ·  ≤7 exp ≤90 DTE</span>"
                    ),
                    font=dict(size=11, color=TEXT2, family="JetBrains Mono"),
                    x=0.01, xanchor="left",
                ),
                height=700,
                margin=dict(t=46, r=210, b=64, l=72),
                hoverlabel=dict(
                    bgcolor=T["bg2"],
                    bordercolor=T["line_bright"],
                    font=dict(family="JetBrains Mono", size=10, color=T["t1"]),
                ),
            )

            # Heatmap x-axis (expiry dates)
            fig.update_xaxes(
                type="category", tickangle=-30,
                tickfont=dict(size=10, family="JetBrains Mono", color=TEXT2),
                showgrid=False,
                linecolor=T["line2"], linewidth=1,
                title=dict(text="Expiration", font=dict(size=9, color=TEXT3)),
                row=1, col=1,
            )
            # Shared y-axis (strike price)
            fig.update_yaxes(
                tickprefix="$",
                tickfont=dict(size=9, family="JetBrains Mono", color=TEXT2),
                range=[spot * 0.953, spot * 1.047],
                gridcolor=T["line2"], gridwidth=1, showgrid=True,
                row=1, col=1,
            )
            # Profile panel x-axis
            fig.update_xaxes(
                tickfont=dict(size=8, family="JetBrains Mono", color=TEXT3),
                showgrid=True, gridcolor=T["line"], gridwidth=1,
                zerolinecolor=T["line2"], zerolinewidth=1,
                title=dict(text="Net GEX", font=dict(size=8, color=TEXT3)),
                row=1, col=2,
            )
            fig.update_yaxes(
                tickprefix="$",
                tickfont=dict(size=9, family="JetBrains Mono", color=TEXT2),
                range=[spot * 0.953, spot * 1.047],
                showgrid=True, gridcolor=T["line"],
                row=1, col=2,
            )
            return fig
        # ── END make_heatmap_fig ──────────────────────────────────────────

        dual = st.session_state.heat_dual
        if dual:
            with st.spinner("Fetching SPY + QQQ heatmap data…"):
                spy_hm  = fetch_options_data_heatmap("SPY")
                qqq_hm  = fetch_options_data_heatmap("QQQ")
            dc1, dc2 = st.columns(2)
            with dc1:
                _spy_df, _spy_spot, _spy_raw = fetch_options_data("SPY", 4)
                if not _spy_df.empty:
                    _gf_spy, _cw_spy, _pw_spy, _ = compute_key_levels(_spy_df, spy_hm[1], _spy_raw)
                else:
                    _gf_spy, _cw_spy, _pw_spy = spy_hm[1], spy_hm[1]*1.01, spy_hm[1]*0.99
                st.plotly_chart(
                    make_heatmap_fig(spy_hm[2], spy_hm[1], "SPY", _gf_spy, _cw_spy, _pw_spy),
                    use_container_width=True, config={"displayModeBar": False}
                )
            with dc2:
                _qqq_df, _qqq_spot, _qqq_raw = fetch_options_data("QQQ", 4)
                if not _qqq_df.empty:
                    _gf_qqq, _cw_qqq, _pw_qqq, _ = compute_key_levels(_qqq_df, qqq_hm[1], _qqq_raw)
                else:
                    _gf_qqq, _cw_qqq, _pw_qqq = qqq_hm[1], qqq_hm[1]*1.01, qqq_hm[1]*0.99
                st.plotly_chart(
                    make_heatmap_fig(qqq_hm[2], qqq_hm[1], "QQQ", _gf_qqq, _cw_qqq, _pw_qqq),
                    use_container_width=True, config={"displayModeBar": False}
                )
        else:
            with st.spinner("Fetching heatmap data…"):
                hm_data = fetch_options_data_heatmap(asset_toggle)
            st.plotly_chart(
                make_heatmap_fig(hm_data[2], hm_data[1], asset_toggle, gamma_flip, call_wall, put_wall),
                use_container_width=True, config={"displayModeBar": False}
            )

    # ── DAILY LEVELS ──────────────────────────────────────────────────────
    elif st.session_state.radar_mode == "DAILY":
        _render_daily_levels(asset_toggle, spot_price, df, raw_df, T)
    # ── EXPECTED MOVE ─────────────────────────────────────────────────────
    elif st.session_state.radar_mode == "MOVE":
        _render_expected_move(asset_toggle, spot_price, df, raw_df, T)
    # ── REPLAY ────────────────────────────────────────────────────────────
    elif st.session_state.radar_mode == "REPLAY":
            _render_replay_view(
                ticker      = asset_toggle,
                spot        = spot_price,
                T           = T,
                gamma_flip  = gamma_flip,
                call_wall   = call_wall,
                put_wall    = put_wall,
                max_pain    = max_pain,
                vol_trigger = vol_trigger,
                df_gex      = df,
                raw_df      = raw_df,
            )
            # ── Save Replay button ────────────────────────────────────────
            _sv_col1, _sv_col2, _ = st.columns([1, 1, 4])
            with _sv_col1:
                _save_label = f"💾 Save {datetime.date.today()} {asset_toggle}"
                if st.button(_save_label, key="save_replay_btn"):
                    with st.spinner("Saving replay to GitHub…"):
                        _save_result = _github_save_replay(
                            ticker   = asset_toggle,
                            date_str = str(datetime.date.today()),
                            T        = T,
                        )
                    if _save_result.get("ok"):
                        st.success(f"Saved! {_save_result.get('msg','')}")
                    else:
                        st.error(f"Save failed: {_save_result.get('err','unknown error')}")
            with _sv_col2:
                st.markdown(
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:9px;'
                    f'color:{T["t3"]};padding-top:10px;">Saves price bars + GEX snapshots to GitHub</div>',
                    unsafe_allow_html=True
                )
    # ── SAVES ────────────────────────────────────────────────────────────
    elif st.session_state.radar_mode == "SAVES":
        _render_saved_replays(T)
    # ── KEY LEVELS TABLE ───────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Key Levels</div>', unsafe_allow_html=True)

    filter_type = st.selectbox(
        "Sort by",
        ["Top Abs GEX","Top Call Walls","Top Put Walls","Closest to Spot"],
        label_visibility="collapsed"
    )

    if filter_type == "Top Abs GEX":
        disp = df.sort_values("abs_gex", ascending=False).head(10)
    elif filter_type == "Top Call Walls":
        disp = df[df["gex_net"]>0].sort_values("gex_net", ascending=False).head(10)
    elif filter_type == "Top Put Walls":
        disp = df[df["gex_net"]<0].sort_values("gex_net", ascending=True).head(10)
    else:
        disp = df.assign(_d=df["strike"].sub(spot_price).abs()).sort_values("_d").head(10).drop(columns=["_d"])

    out = disp[["strike","es_strike","call_gex","put_gex","gex_net",
                "vol_gex_net","dist_pct","open_interest","iv"]].copy()
    out.columns = ["Strike", equiv_label, "Call OI-GEX", "Put OI-GEX",
                   "Net OI-GEX", "Net Vol-GEX", "Dist %", "OI", "IV"]

    def _style(val):
        _ts = get_theme()
        return f"color:{_ts['green'] if val>0 else _ts['red']};font-weight:600;"

    fmt = {"Strike":"{:.1f}", equiv_label:"{:.2f}",
           "Call OI-GEX":"{:.4f}B", "Put OI-GEX":"{:.4f}B",
           "Net OI-GEX":"{:.4f}B", "Net Vol-GEX":"{:.4f}B",
           "Dist %":"{:+.2f}%", "OI":"{:,.0f}", "IV":"{:.3f}"}

    st.dataframe(
        out.style.format(fmt).map(_style, subset=["Net OI-GEX","Net Vol-GEX"]),
        use_container_width=True, hide_index=True,
    )

    # ── Export row ─────────────────────────────────────────────────────────
    csv     = out.to_csv(index=False).encode("utf-8")
    csv_str = out.to_csv(index=False)
    escaped = csv_str.replace("\\","\\\\").replace("`","\\`").replace("$","\\$")

    btn1, btn2 = st.columns([1,1])
    with btn1:
        st.download_button(
            "Export CSV", data=csv,
            file_name=f"{asset_toggle}_gex_levels.csv",
            mime="text/csv", use_container_width=True
        )
    with btn2:
        _components.html(f"""
        <button onclick="
          navigator.clipboard.writeText(`{escaped}`).then(function(){{
            this.innerText='Copied ✓';
            this.style.borderColor=this.dataset.green;
            this.style.color=this.dataset.green;
            this.style.boxShadow='0 0 0 1px rgba(0,229,160,0.25), 0 0 20px rgba(0,229,160,0.1)';
            setTimeout(()=>{{this.innerText='Copy to Clipboard';
              this.style.borderColor='rgba(255,255,255,0.07)';
              this.style.color='#888888';
              this.style.boxShadow='none';
            }},2000);
          }}.bind(this)).catch(()=>prompt('Copy:',`{escaped}`));
        " style="
          width:100%; background:{T['bg']}; border:1px solid {T['line2']}; color:{T['t2']};
          font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:500;
          letter-spacing:1px; text-transform:uppercase; border-radius:2px;
          height:38px; cursor:pointer; transition:all 0.15s ease;
        "
        onmouseover="this.style.borderColor='{T['line_bright']}';this.style.color='{T['t1']}';this.style.background='{T['bg2']}'"
        onmouseout="this.style.borderColor='{T['line2']}';this.style.color='{T['t2']}';this.style.background='{T['bg']}'">
          Copy to Clipboard
        </button>
        """, height=46)


# ── Invoke the fragment ─────────────────────────────────────────────────────
dashboard()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _sb_group(rows_html):
    return f'<div class="sb-group">{rows_html}</div>'

def _m_row(label, value, color="#FFFFFF"):
    return (f'<div class="m-tile">'
            f'<span class="m-label">{label}</span>'
            f'<span class="m-value" style="color:{color}">{value}</span>'
            f'</div>')

_m = st.session_state.get("_sb_metrics", {})

def _v(key, fallback="—"):
    return _m.get(key, fallback)

_spot       = _v("spot_price",     None)
_gflip      = _v("gamma_flip",     None)
_cwall      = _v("call_wall",      None)
_pwall      = _v("put_wall",       None)
_mpain      = _v("max_pain",       None)
_net_gex    = _v("total_net_gex",  None)
_net_vgex   = _v("total_net_vol_gex", None)
_gratio     = _v("gex_ratio",      None)
_vtrig      = _v("vol_trigger",    None)
_mwall      = _v("mom_wall",       None)
_mval       = _v("mom_val",        0.0)
_ivrv       = _v("iv_rv_spread",   None)
_fratio     = _v("flow_ratio",     None)
_nf_str     = _v("nf_str",         "—")
_net_flow   = _v("net_flow",       0.0)

def _fmt(val, fmt_str, prefix="", suffix=""):
    if val is None: return "—"
    return f"{prefix}{val:{fmt_str}}{suffix}"

_SBT = get_theme()
_mom_color = _SBT["blue"] if float(_mval) >= 0 else _SBT["violet"]
_mom_label = "Momentum Wall · Call" if float(_mval) >= 0 else "Momentum Wall · Put"
_mw_str    = f"${_mwall:.2f}" if _mwall else "—"

st.sidebar.markdown("<p class='sb-section'>Market Levels</p>", unsafe_allow_html=True)
st.sidebar.markdown(_sb_group(
    _m_row("Spot",       _fmt(_spot,  ".2f", "$"),  _SBT["green"]) +
    _m_row("Zero Gamma", _fmt(_gflip, ".2f", "$"),  _SBT["t2"]) +
    _m_row("Call Wall",  _fmt(_cwall, ".2f", "$"),  _SBT["green"]) +
    _m_row("Put Wall",   _fmt(_pwall, ".2f", "$"),  _SBT["red"]) +
    _m_row("Max Pain",   _fmt(_mpain, ".2f", "$"),  _SBT["amber"])
), unsafe_allow_html=True)

st.sidebar.markdown("<p class='sb-section'>GEX Exposure</p>", unsafe_allow_html=True)
_gex_col = _SBT["green"] if (_net_gex or 0) > 0 else _SBT["red"]
_vgx_col = _SBT["green"] if (_net_vgex or 0) > 0 else _SBT["red"]
st.sidebar.markdown(_sb_group(
    _m_row("Net OI-GEX",  _fmt(_net_gex,  "+.3f", suffix="B"), _gex_col) +
    _m_row("Net Vol-GEX", _fmt(_net_vgex, "+.3f", suffix="B"), _vgx_col) +
    _m_row("GEX Ratio",   _fmt(_gratio,   ".3f"), _SBT["t1"])
), unsafe_allow_html=True)

st.sidebar.markdown("<p class='sb-section'>Intraday</p>", unsafe_allow_html=True)
st.sidebar.markdown(_sb_group(
    _m_row("Vol Trigger", _fmt(_vtrig, ".2f", "$"),  _SBT["amber"]) +
    _m_row(_mom_label,    _mw_str,                   _mom_color)
), unsafe_allow_html=True)

st.sidebar.markdown("<p class='sb-section'>Volatility</p>", unsafe_allow_html=True)
_iv_rv_color = _SBT["green"] if (_ivrv or 0) > 0 else _SBT["red"]
st.sidebar.markdown(_sb_group(
    _m_row("ATM IV − RV", _fmt(_ivrv, "+.2f", suffix="pp"), _iv_rv_color)
), unsafe_allow_html=True)

st.sidebar.markdown("<p class='sb-section'>Flow</p>", unsafe_allow_html=True)
_nf_col    = _SBT["green"] if float(_net_flow) > 0 else _SBT["red"]
_fr_col    = _SBT["green"] if (_fratio or 0.5) >= 0.5 else _SBT["red"]
st.sidebar.markdown(_sb_group(
    _m_row("Flow Ratio", _fmt(_fratio, ".3f"), _fr_col) +
    _m_row("Net Flow",   _nf_str,              _nf_col)
), unsafe_allow_html=True)