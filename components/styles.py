"""Centralized visual system for the PAIMANA app - palette, CSS, chart theming.

Design direction: "cyanotype ledger" - a blueprint/technical-drawing palette
grounded in the civil-engineering subject matter (MoSPI Flash Reports, cost
ledgers, progress surveys), IBM Plex Sans/Mono (built for technical/enterprise
documentation), and a three-tier card system (instrument tiles, content
panels, status cards) that encodes hierarchy structurally rather than with
uniform rounded-shadow cards.
"""

import base64
from pathlib import Path

import streamlit as st

PALETTE = {
    "bg": "#0A1A2B",
    "surface": "#12283F",
    "surface2": "#1C3A56",
    "text": "#E7F0F6",
    "muted": "#7E97AC",
    "accent": "#6FC3E4",
    "success": "#3FA66B",
    "warning": "#E8A33D",
    "danger": "#D14343",
}

FONTS_DIR = Path("assets/fonts")
FONT_FACES = [
    ("IBM Plex Sans", 400, "ibm-plex-sans-latin-400-normal.woff2"),
    ("IBM Plex Sans", 500, "ibm-plex-sans-latin-500-normal.woff2"),
    ("IBM Plex Sans", 600, "ibm-plex-sans-latin-600-normal.woff2"),
    ("IBM Plex Mono", 400, "ibm-plex-mono-latin-400-normal.woff2"),
    ("IBM Plex Mono", 500, "ibm-plex-mono-latin-500-normal.woff2"),
]


@st.cache_resource
def _font_face_css():
    """Base64-embeds the bundled woff2 files directly into @font-face rules,
    so type renders correctly with no network request - works fully offline
    and avoids any dependency on Streamlit's static-file-serving config."""
    blocks = []
    for family, weight, filename in FONT_FACES:
        data = (FONTS_DIR / filename).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        blocks.append(f"""
        @font-face {{
            font-family: '{family}';
            font-style: normal;
            font-weight: {weight};
            font-display: swap;
            src: url(data:font/woff2;base64,{b64}) format('woff2');
        }}""")
    return "\n".join(blocks)


def inject_global_css():
    st.markdown(f"""
    <style>
    {_font_face_css()}

    .stApp {{ background: {PALETTE['bg']}; font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    code, pre, kbd, samp {{ font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace; }}
    .block-container {{ max-width: 1400px; padding-top: 3.5rem; padding-bottom: 4rem; }}

    /* ---- Hide default streamlit chrome that breaks the premium feel ---- */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    /* Streamlit's fixed header bar sits over the top of the content; give it
       a solid background so it can't visually cut into page titles instead
       of just sitting above them (the block-container padding above is what
       actually reserves the clearance). */
    header[data-testid="stHeader"] {{ background: {PALETTE['bg']}; }}

    h1, h2, h3, h4, p, span, div {{ color: {PALETTE['text']}; }}

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {PALETTE['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {PALETTE['surface2']}; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PALETTE['accent']}88; }}

    /* ---- Focus visibility (accessibility) ---- */
    button:focus-visible, [role="radiogroup"] label:focus-within {{
        outline: 2px solid {PALETTE['accent']}; outline-offset: 2px;
    }}

    /* ---- Instrument status strip (page hero, Overview only) ---- */
    .pm-statusbar {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 24px 28px 0 28px;
        margin-bottom: 24px;
        animation: pm-fadein 0.35s ease;
    }}
    @keyframes pm-fadein {{ from {{ opacity: 0; transform: translateY(-4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .pm-statusbar-title {{ font-size: 1.7rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; }}
    .pm-statusbar-sub {{ color: {PALETTE['muted']}; font-size: 0.95rem; margin: 6px 0 20px 0; }}
    .pm-statusbar-readout {{
        display: flex; flex-wrap: wrap; gap: 32px;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding: 16px 0;
    }}
    .pm-readout-label {{ color: {PALETTE['muted']}; font-size: 0.78rem; font-weight: 500; margin-bottom: 4px; }}
    .pm-readout-value {{
        font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
        font-size: 1.3rem; font-weight: 500; color: {PALETTE['text']};
    }}

    /* ---- Tier 1: instrument tiles (KPI cards) ---- */
    .pm-kpi {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 4px;
        padding: 16px 18px 14px 18px;
        position: relative;
        animation: pm-fadein 0.35s ease;
    }}
    .pm-kpi::before {{
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: {PALETTE['accent']}; border-radius: 4px 4px 0 0;
    }}
    .pm-kpi-label {{ color: {PALETTE['muted']}; font-size: 0.78rem; font-weight: 500; margin-bottom: 6px; }}
    .pm-kpi-value {{
        font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
        font-size: 1.7rem; font-weight: 500;
    }}

    /* ---- Tier 2: content panels ---- */
    .pm-section-title {{
        font-size: 1.2rem; font-weight: 600; margin: 2px 0 8px 0;
        padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .pm-section-sub {{ color: {PALETTE['muted']}; font-size: 0.88rem; margin: 8px 0 14px 0; }}

    /* ---- Native Streamlit bordered container, re-skinned as our panel ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {PALETTE['surface']} !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        transition: border-color 0.18s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: rgba(255,255,255,0.16) !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{ background: transparent !important; }}

    /* ---- Generic panel (kept for single-call HTML snippets) ---- */
    .pm-card {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }}

    /* ---- Risk cards ---- */
    .pm-risk-card {{
        border-radius: 14px; padding: 22px 24px; text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        animation: pm-fadein 0.35s ease;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .pm-risk-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
    }}
    .pm-risk-label {{ color: {PALETTE['muted']}; font-size: 0.8rem; font-weight: 600;
                      letter-spacing: 0.05em; text-transform: uppercase; }}
    .pm-risk-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem; font-weight: 700; margin: 6px 0 2px 0; }}
    .pm-risk-tier {{ font-size: 0.95rem; font-weight: 700; letter-spacing: 0.03em; }}
    .pm-risk-desc {{ color: {PALETTE['muted']}; font-size: 0.85rem; margin-top: 8px; }}

    /* ---- Badges ---- */
    .pm-badge {{
        display: inline-block; padding: 3px 11px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.02em;
    }}

    /* ---- Bar rows for SHAP-style explanations ---- */
    .pm-bar-row {{ margin-bottom: 14px; }}
    .pm-bar-label {{ display: flex; justify-content: space-between; font-size: 0.88rem;
                     margin-bottom: 4px; }}
    .pm-bar-track {{ background: rgba(255,255,255,0.06); border-radius: 6px; height: 8px; width: 100%; }}
    .pm-bar-fill {{ height: 8px; border-radius: 6px; }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{ background: {PALETTE['surface']}; }}
    .pm-nav-brand {{ font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em; margin-bottom: 2px; }}
    .pm-nav-brand-sub {{ color: {PALETTE['muted']}; font-size: 0.72rem; margin-bottom: 18px; }}
    .pm-sidebar-footer {{ color: {PALETTE['muted']}; font-size: 0.72rem; line-height: 1.5;
                          margin-top: 24px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); }}

    /* ---- Sidebar nav (re-skin the bare st.radio into a real nav menu) ---- */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 1px; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 9px 12px; border-radius: 9px; width: 100%;
        transition: background 0.15s ease, color 0.15s ease;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: {PALETTE['accent']}14;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background: {PALETTE['accent']}26;
        box-shadow: inset 3px 0 0 {PALETTE['accent']};
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] input[type="radio"] {{ accent-color: {PALETTE['accent']}; }}

    /* ---- Buttons ---- */
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {{
        border-radius: 8px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    }}
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {{
        transform: translateY(-1px);
    }}
    button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {{
        box-shadow: 0 4px 16px {PALETTE['accent']}55;
    }}

    /* ---- Re-skin native widgets to match the custom card system ---- */
    div[data-testid="stMetric"] {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 14px 18px 10px 18px;
    }}
    div[data-testid="stAlert"] {{ border-radius: 8px; }}
    div[data-testid="stDataFrame"] {{
        border-radius: 8px; overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }}

    /* ---- Footer ---- */
    .pm-footer {{
        color: {PALETTE['muted']}; font-size: 0.78rem; text-align: center;
        margin-top: 48px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.06);
        line-height: 1.6;
    }}

    /* ---- Empty state ---- */
    .pm-empty {{
        color: {PALETTE['muted']}; text-align: center; padding: 36px 20px;
        border: 1px dashed rgba(255,255,255,0.12); border-radius: 8px; font-size: 0.9rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def risk_tier(score):
    """Returns (tier_label, color) for a 0-1 risk PROBABILITY score
    (cost/time/overall risk model outputs)."""
    if score >= 0.5:
        return "HIGH", PALETTE["danger"]
    elif score >= 0.25:
        return "MODERATE", PALETTE["warning"]
    return "LOW", PALETTE["success"]


def cost_overrun_tier(pct, high_threshold, low_threshold=5):
    """Returns (label, color, glyph) for a raw cost-overrun PERCENTAGE
    (distinct scale from risk_tier's 0-1 probability - do not mix the two).
    Shape, not just color, carries the tier so it reads without relying on
    color perception."""
    if pct > high_threshold:
        return "High", PALETTE["danger"], "◆"   # ◆
    if pct > low_threshold:
        return "Moderate", PALETTE["warning"], "▲"  # ▲
    return "Low", PALETTE["success"], "●"   # ●


def style_plotly(fig, height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["text"], size=12),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=height,
    )
    return fig
