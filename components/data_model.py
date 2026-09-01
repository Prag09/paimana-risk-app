"""Data loading + model training. Logic is unchanged from the original app.py -
only extracted into its own module."""

import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import shap
from sklearn.model_selection import cross_val_score

CAT_FEATURES = ["ministry", "state"]
NUM_FEATURES = ["original_cost_cr", "physical_progress_pct"]
COST_THRESHOLD_PCT = 10
TIME_THRESHOLD_MONTHS = 3


def parse_my(series):
    return pd.to_datetime(series, format="%m/%Y", errors="coerce")


@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/projects_master.csv")

    df["cost_overrun_pct"] = (
        (df["revised_cost_cr"] - df["original_cost_cr"]) / df["original_cost_cr"] * 100
    )
    df["cost_at_risk"] = (df["cost_overrun_pct"] > COST_THRESHOLD_PCT).astype(int)

    target_dt = parse_my(df["target_doc"])
    revised_dt = parse_my(df["revised_doc"]).fillna(target_dt)
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

    return {
        "df": df, "X_cols": X, "rcf_baseline": rcf_baseline,
        "cost_model": cost_model, "cost_cv": cost_cv, "cost_explainer": cost_explainer,
        "time_model": time_model, "time_cv": time_cv, "time_explainer": time_explainer,
    }


def predict(state_dict, project_dict):
    """project_dict: {original_cost_cr, physical_progress_pct, ministry, state}
    Returns (cost_score, time_score, cost_factors, time_factors) where *_factors
    is a pandas Series of grouped, human-labeled SHAP contributions."""
    row = pd.DataFrame([project_dict])
    row_encoded = pd.get_dummies(row, columns=CAT_FEATURES).reindex(
        columns=state_dict["X_cols"].columns, fill_value=0
    )

    cost_score = float(state_dict["cost_model"].predict_proba(row_encoded)[0][1])
    time_score = float(state_dict["time_model"].predict_proba(row_encoded)[0][1])

    def grouped_shap(explainer):
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
        friendly = {
            "original_cost_cr": "Project cost", "physical_progress_pct": "Physical progress",
            "ministry": f"Ministry ({project_dict['ministry']})",
            "state": f"State ({project_dict['state']})",
        }
        return pd.Series({friendly.get(k, k): v for k, v in grouped.items()})

    return cost_score, time_score, grouped_shap(state_dict["cost_explainer"]), grouped_shap(state_dict["time_explainer"])
