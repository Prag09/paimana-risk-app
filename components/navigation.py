"""Sidebar navigation for the PAIMANA app."""

import streamlit as st

PAGES = [
    "Overview",
    "Risk Assessment",
    "Search Projects",
    "Regional Intelligence",
    "Trends",
    "Add Project",
    "Methodology",
]


def render_sidebar(n_projects, n_months):
    with st.sidebar:
        st.markdown('<div class="pm-nav-brand">PAIMANA</div>', unsafe_allow_html=True)
        st.markdown('<div class="pm-nav-brand-sub">Infrastructure Risk Intelligence</div>', unsafe_allow_html=True)

        choice = st.radio("Navigation", PAGES, label_visibility="collapsed")

        st.markdown(f"""
        <div class="pm-sidebar-footer">
            <b>SYSTEM STATUS</b><br>
            <span style="color:#32C48D;">●</span> Online<br><br>
            <b>MODEL</b><br>XGBoost · 50 rounds<br><br>
            <b>DATA</b><br>{n_projects} projects · {n_months} month(s)<br><br>
            Team Null Pointers<br>SIH 2026
        </div>
        """, unsafe_allow_html=True)

    return choice
