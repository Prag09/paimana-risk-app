"""Centralized visual system for the PAIMANA app - palette, CSS, chart theming."""

import streamlit as st

PALETTE = {
    "bg": "#080D16",
    "surface": "#111927",
    "surface2": "#182235",
    "text": "#F5F7FA",
    "muted": "#8994A6",
    "accent": "#5B8DEF",
    "success": "#32C48D",
    "warning": "#F3B44B",
    "danger": "#EF6262",
}


def inject_global_css():
    st.markdown(f"""
    <style>
    .stApp {{ background: {PALETTE['bg']}; }}
    .block-container {{ max-width: 1400px; padding-top: 1.5rem; padding-bottom: 4rem; }}

    /* Hide default streamlit chrome that breaks the premium feel */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    h1, h2, h3, h4, p, span, div {{ color: {PALETTE['text']}; }}

    /* ---- Hero ---- */
    .pm-hero {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        display: flex; justify-content: space-between; align-items: flex-start;
    }}
    .pm-hero-title {{ font-size: 2.1rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
    .pm-hero-sub {{ color: {PALETTE['muted']}; font-size: 1rem; margin: 6px 0 0 0; }}
    .pm-hero-tag {{ color: {PALETTE['accent']}; font-size: 0.85rem; font-weight: 600;
                    letter-spacing: 0.04em; margin: 12px 0 0 0; }}
    .pm-status {{
        display: flex; align-items: center; gap: 6px; font-size: 0.8rem;
        color: {PALETTE['success']}; font-weight: 600; white-space: nowrap;
    }}
    .pm-status-dot {{
        width: 8px; height: 8px; border-radius: 50%; background: {PALETTE['success']};
        display: inline-block;
    }}

    /* ---- KPI cards ---- */
    .pm-kpi {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px 20px;
    }}
    .pm-kpi-label {{ color: {PALETTE['muted']}; font-size: 0.78rem; font-weight: 600;
                     letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 6px; }}
    .pm-kpi-value {{ font-size: 1.9rem; font-weight: 700; }}

    /* ---- Section header ---- */
    .pm-section-title {{ font-size: 1.35rem; font-weight: 700; margin: 4px 0 2px 0; }}
    .pm-section-sub {{ color: {PALETTE['muted']}; font-size: 0.92rem; margin-bottom: 16px; }}

    /* ---- Native Streamlit bordered container, re-skinned as our card ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {PALETTE['surface']} !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{ background: transparent !important; }}

    /* ---- Generic card (kept for single-call HTML snippets) ---- */
    .pm-card {{
        background: {PALETTE['surface']};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }}

    /* ---- Risk cards ---- */
    .pm-risk-card {{
        border-radius: 14px; padding: 22px 24px; text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }}
    .pm-risk-label {{ color: {PALETTE['muted']}; font-size: 0.8rem; font-weight: 600;
                      letter-spacing: 0.05em; text-transform: uppercase; }}
    .pm-risk-value {{ font-size: 2.6rem; font-weight: 800; margin: 6px 0 2px 0; }}
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

    /* ---- Footer ---- */
    .pm-footer {{
        color: {PALETTE['muted']}; font-size: 0.78rem; text-align: center;
        margin-top: 48px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.06);
        line-height: 1.6;
    }}

    /* ---- Empty state ---- */
    .pm-empty {{
        color: {PALETTE['muted']}; text-align: center; padding: 36px 20px;
        border: 1px dashed rgba(255,255,255,0.12); border-radius: 12px; font-size: 0.9rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def risk_tier(score):
    """Returns (tier_label, color) for a 0-1 risk score."""
    if score >= 0.5:
        return "HIGH", PALETTE["danger"]
    elif score >= 0.25:
        return "MODERATE", PALETTE["warning"]
    return "LOW", PALETTE["success"]


def style_plotly(fig, height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=PALETTE["text"], size=12),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=height,
    )
    return fig
