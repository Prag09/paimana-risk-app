"""Reusable UI card components for the PAIMANA app."""

import streamlit as st
import plotly.graph_objects as go
from components.styles import PALETTE, risk_tier, style_plotly


def section_header(title, subtitle=None):
    st.markdown(f'<div class="pm-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="pm-section-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value) tuples."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="pm-kpi">
                <div class="pm-kpi-label">{label}</div>
                <div class="pm-kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def risk_badge_html(text, color):
    return f'<span class="pm-badge" style="background:{color}22; color:{color};">{text}</span>'


def risk_card(label, score, description=""):
    tier, color = risk_tier(score)
    st.markdown(f"""
    <div class="pm-risk-card" style="border-color:{color}55; background:{color}0D;">
        <div class="pm-risk-label">{label}</div>
        <div class="pm-risk-value" style="color:{color};">{score*100:.1f}%</div>
        <div class="pm-risk-tier" style="color:{color};">{tier}</div>
        <div class="pm-risk-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def risk_gauge(score, label):
    tier, color = risk_tier(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 30, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PALETTE["muted"], "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(50,196,141,0.10)"},
                {"range": [25, 50], "color": "rgba(243,180,75,0.10)"},
                {"range": [50, 100], "color": "rgba(239,98,98,0.10)"},
            ],
        },
        title={"text": label, "font": {"size": 13, "color": PALETTE["muted"]}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10), height=220,
        font=dict(color=PALETTE["text"]),
    )
    return fig


def factor_bars(factor_series, max_bars=5):
    """factor_series: pandas Series, index=feature label, values=signed SHAP-style contribution."""
    top = factor_series.abs().sort_values(ascending=False).head(max_bars)
    max_val = top.max() if len(top) else 1
    for feat in top.index:
        val = factor_series[feat]
        pct_width = min(100, abs(val) / max_val * 100) if max_val else 0
        color = PALETTE["danger"] if val > 0 else PALETTE["success"]
        direction = "increases risk" if val > 0 else "decreases risk"
        st.markdown(f"""
        <div class="pm-bar-row">
            <div class="pm-bar-label"><span>{feat}</span><span style="color:{color};">{direction}</span></div>
            <div class="pm-bar-track"><div class="pm-bar-fill" style="width:{pct_width:.0f}%; background:{color};"></div></div>
        </div>
        """, unsafe_allow_html=True)


def empty_state(message):
    st.markdown(f'<div class="pm-empty">{message}</div>', unsafe_allow_html=True)


def styled_dataframe(df, rename=None, pct_cols=None, currency_cols=None, use_container_width=True):
    """Light formatting wrapper around st.dataframe for a less 'raw pandas' look."""
    d = df.copy()
    if rename:
        d = d.rename(columns=rename)
    col_config = {}
    if pct_cols:
        for c in pct_cols:
            label = rename.get(c, c) if rename else c
            col_config[label] = st.column_config.NumberColumn(label, format="%.1f%%")
    if currency_cols:
        for c in currency_cols:
            label = rename.get(c, c) if rename else c
            col_config[label] = st.column_config.NumberColumn(label, format="₹%.1f Cr")
    st.dataframe(d, use_container_width=use_container_width, hide_index=True, column_config=col_config)
