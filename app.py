"""
PAIMANA Risk & Insights Platform (v2)
========================================
Run with: streamlit run app.py
Reads data/projects_master.csv (grows as you run ingest_report.py on more months).
"""

import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import shap
import plotly.express as px
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="PAIMANA Risk & Insights", layout="wide", page_icon="🏗️")

CAT_FEATURES = ["ministry", "state"]
NUM_FEATURES = ["original_cost_cr", "physical_progress_pct"]
COST_THRESHOLD_PCT = 10       # cost overrun beyond this % = flagged
TIME_THRESHOLD_MONTHS = 3     # schedule slip beyond this many months = flagged

# ---------------------------------------------------------------
# Visual polish: custom CSS on top of the theme
# ---------------------------------------------------------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1B3A5C 0%, #2E86AB 100%);
    padding: 28px 32px; border-radius: 14px; margin-bottom: 18px;
}
.hero h1 { color: white; margin: 0; font-size: 2rem; }
.hero p { color: #DCE8F5; margin: 4px 0 0 0; }
div[data-testid="stMetric"] {
    background: #182238; border: 1px solid #2A3A55;
    border-radius: 10px; padding: 12px 16px;
}
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; margin-right: 6px;
}
.badge-red { background: #4A1F1F; color: #FF8080; }
.badge-yellow { background: #4A3E1F; color: #FFD166; }
.badge-green { background: #1F4A2A; color: #7CE38B; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------
def parse_my(series):
    return pd.to_datetime(series, format="%m/%Y", errors="coerce")


@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/projects_master.csv")

    # --- cost overrun (already present) ---
    df["cost_overrun_pct"] = (
        (df["revised_cost_cr"] - df["original_cost_cr"]) / df["original_cost_cr"] * 100
    )
    df["cost_at_risk"] = (df["cost_overrun_pct"] > COST_THRESHOLD_PCT).astype(int)

    # --- time overrun (new) ---
    target_dt = parse_my(df["target_doc"])
    revised_dt = parse_my(df["revised_doc"]).fillna(target_dt)  # not-yet-revised = 0 slip so far
    df["time_overrun_months"] = (
        (revised_dt.dt.year - target_dt.dt.year) * 12 + (revised_dt.dt.month - target_dt.dt.month)
    )
    df["time_at_risk"] = (df["time_overrun_months"] > TIME_THRESHOLD_MONTHS).astype(int)

    rcf_baseline = df.groupby("ministry").agg(
        cost_median=("cost_overrun_pct", "median"),
        time_median=("time_overrun_months", "median"),
        count=("cost_overrun_pct", "count"),
    )

    X = pd.get_dummies(df[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES)

    def train_model(y):
        m = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, reg_lambda=2.0, eval_metric="logloss", random_state=42,
        )
        m.fit(X, y)
        try:
            cv = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
        except ValueError:
            cv = np.array([np.nan])
        return m, cv

    cost_model, cost_cv = train_model(df["cost_at_risk"])
    time_model, time_cv = train_model(df["time_at_risk"])

    cost_explainer = shap.TreeExplainer(cost_model)
    time_explainer = shap.TreeExplainer(time_model)

    return df, X, rcf_baseline, cost_model, cost_cv, cost_explainer, time_model, time_cv, time_explainer


(df, X_train_cols, rcf_baseline, cost_model, cost_cv, cost_explainer,
 time_model, time_cv, time_explainer) = load_and_train()

n_months = df["report_month"].nunique()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🏗️ PAIMANA Risk & Insights Platform</h1>
  <p>SIH26103 · {len(df)} real project records · {n_months} month(s) of MoSPI Flash Report data</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔮 Predict & Summarize", "📍 Regional History", "⏱️ Time & Cost Trends",
     "🌦️ Seasonal Patterns", "ℹ️ About & Model Details"]
)


def risk_badge(score, label):
    if score >= 0.5:
        cls, tag = "badge-red", "HIGH RISK"
    elif score >= 0.25:
        cls, tag = "badge-yellow", "MODERATE"
    else:
        cls, tag = "badge-green", "LOW RISK"
    st.markdown(
        f'<span class="badge {cls}">{label}: {score*100:.1f}% — {tag}</span>',
        unsafe_allow_html=True
    )


def top_shap_factors(row_encoded, explainer, n=3):
    shap_vals = explainer.shap_values(row_encoded)[0]
    contributions = pd.Series(shap_vals, index=row_encoded.columns)
    grouped = {}
    for col, val in contributions.items():
        base = col
        for cf in CAT_FEATURES:
            if col.startswith(cf + "_"):
                base = cf
                break
        grouped[base] = grouped.get(base, 0) + val
    grouped = pd.Series(grouped)
    return grouped.abs().sort_values(ascending=False).head(n), grouped


# ============================================================
# TAB 1: Predict & Summarize
# ============================================================
with tab1:
    st.subheader("Enter project details")
    col1, col2 = st.columns(2)
    with col1:
        ministry = st.selectbox("Ministry", sorted(df["ministry"].dropna().unique()))
        original_cost = st.number_input("Original approved cost (Rs. Crore)", min_value=1.0, value=500.0, step=10.0)
    with col2:
        state = st.selectbox("State", sorted(df["state"].dropna().unique()))
        progress = st.slider("Current physical progress (%)", 0, 100, 40)

    if st.button("Predict risk", type="primary"):
        row = pd.DataFrame([{
            "original_cost_cr": original_cost, "physical_progress_pct": progress,
            "ministry": ministry, "state": state,
        }])
        row_encoded = pd.get_dummies(row, columns=CAT_FEATURES).reindex(columns=X_train_cols.columns, fill_value=0)

        cost_score = float(cost_model.predict_proba(row_encoded)[0][1])
        time_score = float(time_model.predict_proba(row_encoded)[0][1])

        st.divider()
        risk_badge(cost_score, "Cost overrun risk")
        risk_badge(time_score, "Schedule slip risk")
        st.caption("Cost: probability of exceeding 10% cost overrun. "
                    "Schedule: probability of slipping more than 3 months. Based on similar past projects.")

        top3, grouped = top_shap_factors(row_encoded, cost_explainer)
        friendly = {
            "original_cost_cr": "Project cost", "physical_progress_pct": "Physical progress",
            "ministry": f"Ministry ({ministry})", "state": f"State ({state})",
        }
        st.markdown("**Why (cost risk):**")
        for feat in top3.index:
            direction = "increases" if grouped[feat] > 0 else "decreases"
            st.write(f"- **{friendly.get(feat, feat)}** {direction} risk")

        # ----- Rich summary with actual past-project table -----
        st.markdown("### 📋 Past experience for this ministry + state")
        similar = df[(df["ministry"] == ministry) & (df["state"] == state)]
        scope_note = f"{ministry} projects in {state}"
        if len(similar) == 0:
            similar = df[df["ministry"] == ministry]
            scope_note = f"{ministry} projects (all states — none found specifically in {state})"

        if len(similar) > 0:
            avg_cost_overrun = similar["cost_overrun_pct"].mean()
            avg_time_overrun = similar["time_overrun_months"].mean()
            pct_cost_risk = (similar["cost_overrun_pct"] > COST_THRESHOLD_PCT).mean() * 100
            pct_time_risk = (similar["time_overrun_months"] > TIME_THRESHOLD_MONTHS).mean() * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Similar projects found", len(similar))
            c2.metric("Avg cost overrun", f"{avg_cost_overrun:.1f}%")
            c3.metric("Avg schedule slip", f"{avg_time_overrun:.1f} mo")
            c4.metric("% historically flagged risky", f"{max(pct_cost_risk, pct_time_risk):.0f}%")

            st.write(
                f"Based on **{len(similar)}** past {scope_note}: average cost overrun was "
                f"**{avg_cost_overrun:.1f}%**, average schedule slip was **{avg_time_overrun:.1f} months**. "
                f"**{pct_cost_risk:.0f}%** exceeded the cost-risk threshold and **{pct_time_risk:.0f}%** "
                f"exceeded the schedule-risk threshold. This is a pattern from similar past projects, "
                f"not a guaranteed outcome for this specific project."
            )
            if len(similar) < 5:
                st.caption("⚠️ Small sample — treat this as low-confidence.")

            st.dataframe(
                similar[["project_name", "state", "original_cost_cr", "revised_cost_cr",
                         "cost_overrun_pct", "time_overrun_months", "physical_progress_pct"]]
                .rename(columns={
                    "project_name": "Project", "state": "State",
                    "original_cost_cr": "Original cost (Cr)", "revised_cost_cr": "Revised cost (Cr)",
                    "cost_overrun_pct": "Cost overrun %", "time_overrun_months": "Time slip (mo)",
                    "physical_progress_pct": "Progress %",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No comparable historical projects found yet for this combination.")

# ============================================================
# TAB 2: Regional History
# ============================================================
with tab2:
    st.subheader("All past projects in a region")
    region = st.selectbox("Select state", sorted(df["state"].dropna().unique()), key="region_select")
    region_df = df[df["state"] == region]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total projects", len(region_df))
    c2.metric("Avg cost overrun", f"{region_df['cost_overrun_pct'].mean():.1f}%")
    c3.metric("Avg schedule slip", f"{region_df['time_overrun_months'].mean():.1f} mo")
    c4.metric("% flagged at-risk", f"{(region_df['cost_overrun_pct'] > COST_THRESHOLD_PCT).mean()*100:.0f}%")

    fig = px.histogram(region_df, x="cost_overrun_pct", nbins=20,
                        title=f"Cost overrun distribution — {region}",
                        labels={"cost_overrun_pct": "Cost overrun %"},
                        color_discrete_sequence=["#2E86AB"])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        region_df[["project_name", "ministry", "original_cost_cr", "revised_cost_cr",
                   "cost_overrun_pct", "time_overrun_months", "physical_progress_pct"]],
        use_container_width=True, hide_index=True,
    )

# ============================================================
# TAB 3: Time & Cost Trends
# ============================================================
with tab3:
    st.subheader("How cost and schedule overrun relate to progress")
    colA, colB = st.columns(2)
    with colA:
        fig1 = px.scatter(df, x="physical_progress_pct", y="cost_overrun_pct", color="ministry",
                           title="Cost overrun % vs physical progress %",
                           labels={"physical_progress_pct": "Physical progress %",
                                   "cost_overrun_pct": "Cost overrun %"})
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        fig1b = px.scatter(df, x="physical_progress_pct", y="time_overrun_months", color="ministry",
                            title="Schedule slip (months) vs physical progress %",
                            labels={"physical_progress_pct": "Physical progress %",
                                    "time_overrun_months": "Schedule slip (months)"})
        st.plotly_chart(fig1b, use_container_width=True)

    st.caption(
        "⚠️ Note: this correlation partly reflects reporting lag, not pure causation — "
        "early-stage projects haven't had time to reveal cost/schedule revisions yet. See About tab."
    )

    st.subheader("Overrun spread by ministry")
    fig2 = px.box(df, x="ministry", y="cost_overrun_pct", title="Cost overrun spread by ministry",
                  color_discrete_sequence=["#2E86AB"])
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB 4: Seasonal Patterns
# ============================================================
with tab4:
    st.subheader("Does approval timing correlate with overrun?")
    seasonal_df = df.copy()
    seasonal_df["approval_month"] = parse_my(seasonal_df["approval_date"]).dt.month
    monthly = seasonal_df.groupby("approval_month")[["cost_overrun_pct", "time_overrun_months"]].mean().reindex(range(1, 13))

    fig3 = px.bar(
        x=[f"{m:02d}" for m in monthly.index], y=monthly["cost_overrun_pct"].values,
        labels={"x": "Approval month", "y": "Avg cost overrun %"},
        title="Average cost overrun by approval month (proxy for seasonal effects incl. monsoon)",
        color_discrete_sequence=["#2E86AB"],
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Approval month is a rough proxy for season — for a stronger signal, join actual "
        "monsoon/rainfall data by state and month. Flagged as a next step, not done yet."
    )

# ============================================================
# TAB 5: About
# ============================================================
with tab5:
    st.subheader("Model & validation details")
    cost_cv_txt = f"{cost_cv.mean():.3f} (± {cost_cv.std():.3f})" if not np.isnan(cost_cv).all() else "N/A"
    time_cv_txt = f"{time_cv.mean():.3f} (± {time_cv.std():.3f})" if not np.isnan(time_cv).all() else "N/A"
    st.markdown(f"""
    **Algorithm:** XGBoost (Gradient Boosted Decision Trees), 50 boosting rounds, max depth 3,
    trained via scikit-learn's `.fit()` API. Two separate models: one for cost-overrun risk,
    one for schedule-slip risk — same input features (cost, progress, ministry, state).

    **Validation:** 5-fold cross-validation — the dataset is split 5 different ways, each model
    trained on 4/5 and tested on the held-out 1/5, five independent times, then a final model is
    fit on all data. This means the reported score isn't from one lucky split.

    - Cost-risk model 5-fold ROC-AUC: **{cost_cv_txt}**
    - Schedule-risk model 5-fold ROC-AUC: **{time_cv_txt}**

    **Explainability:** SHAP (TreeExplainer) attributes each prediction to its driving features.

    **Data:** {len(df)} project records across {n_months} month(s) of real MoSPI PAIMANA
    Flash Report data, extracted via automated PDF parsing.

    **Known limitations (say these out loud in the demo — it builds credibility, not weakness):**
    - Physical progress correlates with recorded risk partly due to reporting lag, not pure causation.
    - Small sample size per ministry/state combination — treat low-count predictions as low-confidence.
    - Seasonal analysis currently uses approval month as a rough proxy; true monsoon/rainfall data
      would strengthen this.
    """)
