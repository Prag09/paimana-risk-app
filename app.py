"""
PAIMANA — Infrastructure Risk Intelligence
=============================================
Run with: streamlit run app.py
All ML/data logic lives in components/data_model.py, unchanged from the
original single-file app. This file is UI orchestration only.
"""

import pandas as pd
import streamlit as st

from components.styles import inject_global_css, PALETTE
from components.cards import (
    section_header, kpi_row, risk_card, risk_gauge, factor_bars,
    empty_state, styled_dataframe,
)
from components.charts import (
    risk_distribution_donut, cost_overrun_scatter, time_overrun_scatter,
    ministry_box, region_bar, seasonal_bar, histogram,
)
from components.navigation import render_sidebar
from components.data_model import load_and_train, predict, parse_my, COST_THRESHOLD_PCT, TIME_THRESHOLD_MONTHS

st.set_page_config(page_title="PAIMANA — Infrastructure Risk Intelligence", layout="wide", page_icon="🛰️")
inject_global_css()

state = load_and_train()
df = state["df"]
n_months = df["report_month"].nunique()

if "added_rows" not in st.session_state:
    st.session_state.added_rows = pd.DataFrame(columns=df.columns)

full_df = pd.concat([df, st.session_state.added_rows], ignore_index=True) if len(st.session_state.added_rows) else df

page = render_sidebar(len(df), n_months)

# ============================================================
# HERO (shown on every page, lightweight)
# ============================================================
st.markdown(f"""
<div class="pm-hero">
  <div>
    <div class="pm-hero-title">PAIMANA</div>
    <div class="pm-hero-sub">Infrastructure Risk Intelligence — AI-assisted risk assessment for infrastructure projects</div>
    <div class="pm-hero-tag">PREDICT · EXPLAIN · MONITOR</div>
  </div>
  <div class="pm-status"><span class="pm-status-dot"></span> SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: OVERVIEW — "What is happening?"
# ============================================================
if page == "Overview":
    high_risk_pct = (df["cost_overrun_pct"] > COST_THRESHOLD_PCT).mean() * 100
    total_value = df["original_cost_cr"].sum()
    kpi_row([
        ("Projects Analysed", f"{len(df):,}"),
        ("States Covered", f"{df['state'].nunique()}"),
        ("Portfolio Value", f"₹{total_value/1000:,.1f}k Cr"),
        ("High Risk Projects", f"{high_risk_pct:.0f}%"),
    ])
    st.write("")

    col1, col2 = st.columns([1, 1.3])
    with col1:
        with st.container(border=True):
            section_header("Risk Distribution", "Share of portfolio by cost-overrun risk tier")
            low = (df["cost_overrun_pct"] <= 5).mean() * 100
            high = high_risk_pct
            moderate = 100 - low - high
            st.plotly_chart(risk_distribution_donut(round(low), round(moderate), round(high)),
                             width='stretch', config={"displayModeBar": False})

    with col2:
        with st.container(border=True):
            section_header("Top Risk Drivers", "Aggregate feature importance from the cost-risk model")
            importances = pd.Series(
                state["cost_model"].feature_importances_, index=state["X_cols"].columns
            )
            grouped = {}
            for col, val in importances.items():
                base = col
                for cf in ["ministry", "state"]:
                    if col.startswith(cf + "_"):
                        base = cf.capitalize()
                        break
                else:
                    base = {"original_cost_cr": "Project Cost", "physical_progress_pct": "Physical Progress"}.get(col, col)
                grouped[base] = grouped.get(base, 0) + val
            factor_bars(pd.Series(grouped), max_bars=4)

    st.write("")
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section_header("Regional Exposure", "Average cost overrun by state (top 8 by project count)")
            top_states = df["state"].value_counts().head(8).index
            region_summary = df[df["state"].isin(top_states)].groupby("state", as_index=False)["cost_overrun_pct"].mean()
            st.plotly_chart(region_bar(region_summary, "cost_overrun_pct", "Avg cost overrun %"),
                             width='stretch', config={"displayModeBar": False})

    with col4:
        with st.container(border=True):
            section_header("Recent Risk Signals", "Highest cost-overrun projects currently on record")
            signals = df.sort_values("cost_overrun_pct", ascending=False).head(6)
            styled_dataframe(
                signals[["project_name", "state", "cost_overrun_pct"]],
                rename={"project_name": "Project", "state": "State", "cost_overrun_pct": "Overrun"},
                pct_cols=["cost_overrun_pct"],
            )


# ============================================================
# PAGE: RISK ASSESSMENT — "How risky is this project?"
# ============================================================
elif page == "Risk Assessment":
    section_header("Risk Assessment", "Assess project-level cost and schedule risk")

    with st.container(border=True):
        st.markdown("**PROJECT DETAILS**")
        c1, c2 = st.columns(2)
        with c1:
            ministry = st.selectbox("Ministry / Sector", sorted(df["ministry"].dropna().unique()))
            original_cost = st.number_input("Original Cost (₹ Crore)", min_value=1.0, value=500.0, step=10.0)
        with c2:
            proj_state = st.selectbox("State", sorted(df["state"].dropna().unique()))
            progress = st.slider("Current Physical Progress (%)", 0, 100, 40)
        run = st.button("Analyze Project →", type="primary")

    if run:
        with st.spinner("Evaluating historical patterns · running risk model · generating explanation..."):
            cost_score, time_score, cost_factors, time_factors = predict(
                state, {"original_cost_cr": original_cost, "physical_progress_pct": progress,
                        "ministry": ministry, "state": proj_state}
            )

        st.write("")
        rc1, rc2 = st.columns(2)
        with rc1:
            risk_card("Cost Risk", cost_score, f"Probability of exceeding {COST_THRESHOLD_PCT}% cost overrun")
        with rc2:
            risk_card("Schedule Risk", time_score, f"Probability of slipping more than {TIME_THRESHOLD_MONTHS} months")

        st.write("")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(risk_gauge(cost_score, "COST RISK"), width='stretch', config={"displayModeBar": False})
        with g2:
            st.plotly_chart(risk_gauge(time_score, "SCHEDULE RISK"), width='stretch', config={"displayModeBar": False})

        with st.container(border=True):
            section_header("Why is this project flagged?", "SHAP feature attribution — cost-risk model")
            factor_bars(cost_factors)
            top_driver = cost_factors.abs().sort_values(ascending=False).index[0]
            st.markdown(f"""
            <div style="margin-top:10px; padding:14px 16px; background:{PALETTE['surface2']}; border-radius:10px;">
                <div style="color:{PALETTE['muted']}; font-size:0.75rem; font-weight:700; letter-spacing:0.04em;">PRIMARY RISK DRIVER</div>
                <div style="margin-top:4px;">{top_driver} is the strongest contributor to this prediction.</div>
            </div>
            """, unsafe_allow_html=True)

        with st.container(border=True):
            section_header("Similar Projects", "Historical projects comparable to this assessment")
            similar = df[(df["ministry"] == ministry) & (df["state"] == proj_state)]
            scope = f"{ministry} projects in {proj_state}"
            if len(similar) == 0:
                similar = df[df["ministry"] == ministry]
                scope = f"{ministry} projects (all states)"

            if len(similar) > 0:
                avg_cost = similar["cost_overrun_pct"].mean()
                avg_time = similar["time_overrun_months"].mean()
                kpi_row([
                    ("Similar Projects", f"{len(similar)}"),
                    ("Avg Cost Overrun", f"{avg_cost:.1f}%"),
                    ("Avg Schedule Delay", f"{avg_time:.1f} mo"),
                ])
                st.caption(f"Based on {len(similar)} past {scope}. Pattern from history, not a guarantee.")
                if len(similar) < 5:
                    st.caption("⚠️ Small sample — treat as low-confidence.")
                styled_dataframe(
                    similar[["project_name", "state", "original_cost_cr", "revised_cost_cr", "cost_overrun_pct", "time_overrun_months"]],
                    rename={"project_name": "Project", "state": "State", "original_cost_cr": "Original Cost",
                            "revised_cost_cr": "Revised Cost", "cost_overrun_pct": "Overrun", "time_overrun_months": "Slip (mo)"},
                    pct_cols=["cost_overrun_pct"], currency_cols=["original_cost_cr", "revised_cost_cr"],
                )
            else:
                empty_state("No comparable historical projects found yet for this combination.")
    else:
        empty_state("Enter project details above and click Analyze Project to see risk results.")


# ============================================================
# PAGE: SEARCH PROJECTS
# ============================================================
elif page == "Search Projects":
    section_header("Search Projects", "Find projects by name, region, ministry, cost, or reporting month")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            query = st.text_input("Search by project or agency name")
            ministries = st.multiselect("Ministry", sorted(full_df["ministry"].dropna().unique()))
        with c2:
            states_f = st.multiselect("State", sorted(full_df["state"].dropna().unique()))
            months_f = st.multiselect("Report month", sorted(full_df["report_month"].dropna().unique()))
        with c3:
            cost_range = st.slider("Original cost range (₹ Cr)", 0, int(full_df["original_cost_cr"].max()) + 1,
                                    (0, int(full_df["original_cost_cr"].max()) + 1))

    results = full_df.copy()
    if query:
        mask = results["project_name"].str.contains(query, case=False, na=False)
        if "agency" in results.columns:
            mask = mask | results["agency"].str.contains(query, case=False, na=False)
        results = results[mask]
    if ministries:
        results = results[results["ministry"].isin(ministries)]
    if states_f:
        results = results[results["state"].isin(states_f)]
    if months_f:
        results = results[results["report_month"].isin(months_f)]
    results = results[(results["original_cost_cr"] >= cost_range[0]) & (results["original_cost_cr"] <= cost_range[1])]

    with st.container(border=True):
        section_header(f"{len(results)} project(s) found")
        if len(results) > 0:
            styled_dataframe(
                results[["project_name", "ministry", "state", "original_cost_cr", "cost_overrun_pct",
                         "time_overrun_months", "physical_progress_pct", "report_month"]],
                rename={"project_name": "Project", "ministry": "Ministry", "state": "State",
                        "original_cost_cr": "Cost", "cost_overrun_pct": "Overrun",
                        "time_overrun_months": "Slip (mo)", "physical_progress_pct": "Progress %",
                        "report_month": "Report Month"},
                pct_cols=["cost_overrun_pct"], currency_cols=["original_cost_cr"],
            )
        else:
            empty_state("No projects match these filters. Try widening your search.")


# ============================================================
# PAGE: REGIONAL INTELLIGENCE — "Where is risk concentrated?"
# ============================================================
elif page == "Regional Intelligence":
    section_header("Regional Intelligence", "Understand risk exposure across regions and states")

    state_summary = df.groupby("state").agg(
        projects=("project_name", "count"),
        avg_overrun=("cost_overrun_pct", "mean"),
    ).reset_index()
    high_risk_states = (state_summary["avg_overrun"] > COST_THRESHOLD_PCT).sum()
    worst_state = state_summary.sort_values("avg_overrun", ascending=False).iloc[0]

    kpi_row([
        ("High-Risk States", f"{high_risk_states}"),
        ("Avg Regional Overrun", f"{state_summary['avg_overrun'].mean():.1f}%"),
        ("Projects Analysed", f"{len(df)}"),
        ("Highest-Risk Region", worst_state["state"]),
    ])
    st.write("")

    with st.container(border=True):
        section_header("Risk Exposure by Region", "Average cost overrun across all states")
        st.plotly_chart(region_bar(state_summary.sort_values("avg_overrun", ascending=False),
                                    "avg_overrun", "Avg cost overrun %"),
                         width='stretch', config={"displayModeBar": False})

    with st.container(border=True):
        section_header("State Comparison", "Select a state to view its project history")
        region = st.selectbox("State", sorted(df["state"].dropna().unique()))
        region_df = df[df["state"] == region]
        kpi_row([
            ("Total Projects", f"{len(region_df)}"),
            ("Avg Cost Overrun", f"{region_df['cost_overrun_pct'].mean():.1f}%"),
            ("Avg Schedule Slip", f"{region_df['time_overrun_months'].mean():.1f} mo"),
            ("% Flagged At-Risk", f"{(region_df['cost_overrun_pct'] > COST_THRESHOLD_PCT).mean()*100:.0f}%"),
        ])
        st.plotly_chart(histogram(region_df, "cost_overrun_pct", "Cost overrun %"),
                         width='stretch', config={"displayModeBar": False})
        styled_dataframe(
            region_df[["project_name", "ministry", "original_cost_cr", "cost_overrun_pct", "physical_progress_pct"]],
            rename={"project_name": "Project", "ministry": "Ministry", "original_cost_cr": "Cost",
                    "cost_overrun_pct": "Overrun", "physical_progress_pct": "Progress %"},
            pct_cols=["cost_overrun_pct"], currency_cols=["original_cost_cr"],
        )

    with st.container(border=True):
        section_header("Regional Leaders & Laggards", "Best and worst average performance, states with 3+ projects")
        eligible = state_summary[state_summary["projects"] >= 3]
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Best performing**")
            styled_dataframe(eligible.sort_values("avg_overrun").head(5),
                              rename={"state": "State", "projects": "Projects", "avg_overrun": "Avg Overrun"},
                              pct_cols=["avg_overrun"])
        with l2:
            st.markdown("**Highest risk**")
            styled_dataframe(eligible.sort_values("avg_overrun", ascending=False).head(5),
                              rename={"state": "State", "projects": "Projects", "avg_overrun": "Avg Overrun"},
                              pct_cols=["avg_overrun"])


# ============================================================
# PAGE: TRENDS — "How is risk changing?"
# ============================================================
elif page == "Trends":
    section_header("Trends", "Cost, schedule, and seasonal patterns across the portfolio")

    with st.container(border=True):
        section_header("Cost Overrun vs. Progress")
        st.plotly_chart(cost_overrun_scatter(df), width='stretch', config={"displayModeBar": False})
        st.caption("⚠️ Correlation partly reflects reporting lag, not pure causation — see Methodology.")

    with st.container(border=True):
        section_header("Schedule Slip vs. Progress")
        st.plotly_chart(time_overrun_scatter(df), width='stretch', config={"displayModeBar": False})

    with st.container(border=True):
        section_header("Cost Overrun Spread by Ministry")
        st.plotly_chart(ministry_box(df), width='stretch', config={"displayModeBar": False})

    with st.container(border=True):
        section_header("Seasonal Patterns", "Average cost overrun by approval month (proxy for seasonal effects)")
        seasonal_df = df.copy()
        seasonal_df["approval_month"] = parse_my(seasonal_df["approval_date"]).dt.month
        monthly = seasonal_df.groupby("approval_month")[["cost_overrun_pct", "time_overrun_months"]].mean().reindex(range(1, 13))
        st.plotly_chart(seasonal_bar(monthly, "cost_overrun_pct", "Avg cost overrun %"),
                         width='stretch', config={"displayModeBar": False})
        st.caption("Approval month is a rough seasonal proxy — true monsoon/rainfall data would strengthen this.")


# ============================================================
# PAGE: ADD PROJECT
# ============================================================
elif page == "Add Project":
    section_header("Add Project", "Add a project to this session's dataset")
    st.info("⚠️ Additions here are session-only on Streamlit Cloud's free tier — they reset on app restart. "
            "Download your additions below and fold them into projects_master.csv permanently if needed.")

    with st.container(border=True):
        with st.form("add_project_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_ministry = st.selectbox("Ministry", sorted(df["ministry"].dropna().unique()))
                new_state = st.selectbox("State", sorted(df["state"].dropna().unique()))
                new_name = st.text_input("Project name")
            with c2:
                new_orig_cost = st.number_input("Original cost (₹ Cr)", min_value=1.0, value=100.0)
                new_rev_cost = st.number_input("Revised cost (₹ Cr)", min_value=1.0, value=100.0)
                new_progress = st.slider("Physical progress (%)", 0, 100, 0)
            submitted = st.form_submit_button("Add to session dataset")

        if submitted:
            if not new_name.strip():
                st.error("Project name is required.")
            else:
                new_row = pd.DataFrame([{
                    "ministry": new_ministry, "state": new_state, "project_name": new_name,
                    "original_cost_cr": new_orig_cost, "revised_cost_cr": new_rev_cost,
                    "physical_progress_pct": new_progress,
                    "cost_overrun_pct": (new_rev_cost - new_orig_cost) / new_orig_cost * 100,
                    "time_overrun_months": 0, "report_month": "session-added",
                }])
                st.session_state.added_rows = pd.concat([st.session_state.added_rows, new_row], ignore_index=True)
                st.success(f"Added '{new_name}' to this session. It now appears in Search Projects.")

    if len(st.session_state.added_rows) > 0:
        with st.container(border=True):
            section_header("Projects added this session")
            st.dataframe(st.session_state.added_rows, width='stretch', hide_index=True)
            st.download_button(
                "Download session additions as CSV",
                st.session_state.added_rows.to_csv(index=False),
                "session_additions.csv", "text/csv",
            )


# ============================================================
# PAGE: METHODOLOGY — "How does the system work?"
# ============================================================
elif page == "Methodology":
    section_header("Methodology", "How PAIMANA's risk model works")

    steps = [
        ("01", "DATA", "Historical infrastructure project data extracted from real MoSPI PAIMANA Flash Reports."),
        ("02", "FEATURES", "Cost, physical progress, ministry, and state — the features consistently available across all reports."),
        ("03", "MODEL", "XGBoost (Gradient Boosted Decision Trees) — two separate classifiers, one for cost risk, one for schedule risk."),
        ("04", "EXPLAINABILITY", "SHAP (TreeExplainer) attributes each individual prediction to its driving features."),
        ("05", "OUTPUTS", "Cost Risk · Schedule Risk · Risk Drivers · Historical Context, per project."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="pm-card" style="display:flex; gap:20px; align-items:flex-start;">
            <div style="font-size:1.6rem; font-weight:800; color:{PALETTE['accent']}; min-width:44px;">{num}</div>
            <div><b>{title}</b><div style="color:{PALETTE['muted']}; margin-top:4px;">{desc}</div></div>
        </div>
        """, unsafe_allow_html=True)

    cost_cv_txt = f"{state['cost_cv'].mean():.3f} (± {state['cost_cv'].std():.3f})"
    time_cv_txt = f"{state['time_cv'].mean():.3f} (± {state['time_cv'].std():.3f})"
    with st.container(border=True):
        section_header("Validated Model Performance", "5-fold cross-validation ROC-AUC — 5 independent train/test splits per model")
        kpi_row([
            ("Cost-Risk Model AUC", cost_cv_txt),
            ("Schedule-Risk Model AUC", time_cv_txt),
            ("Training Data", f"{len(df)} rows / {n_months} mo"),
        ])

    st.markdown(f"""
    <div class="pm-card" style="border-left: 3px solid {PALETTE['accent']};">
        <b>MODEL INTERPRETATION</b>
        <div style="color:{PALETTE['muted']}; margin-top:8px; line-height:1.6;">
        Predictions are decision-support signals, not deterministic outcomes. Risk estimates depend on
        historical data and the quality of available project features.<br><br>
        <b>Known limitations:</b><br>
        • Physical progress correlates with recorded risk partly due to reporting lag — a project without
        a recorded revision isn't necessarily safe, it may simply not have been revised yet.<br>
        • Small sample size for some ministry/state combinations — treat low-count predictions as low-confidence.<br>
        • Seasonal analysis uses approval month as a rough proxy; true monsoon/rainfall data would strengthen it.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="pm-footer">
    <b>PAIMANA</b> — Infrastructure Risk Intelligence<br>
    AI-assisted decision support for infrastructure project monitoring<br><br>
    Developed by Team Null Pointers · Smart India Hackathon 2026
</div>
""", unsafe_allow_html=True)
