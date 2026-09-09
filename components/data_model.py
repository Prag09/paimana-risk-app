"""Data loading + model training. Logic is unchanged from the original app.py -
only extracted into its own module."""

import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import shap
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

CAT_FEATURES = ["ministry", "state"]
NUM_FEATURES = ["original_cost_cr", "physical_progress_pct"]
COST_THRESHOLD_PCT = 10
TIME_THRESHOLD_MONTHS = 3
CONFIDENCE_LEVEL = 0.90  # for conformal prediction intervals
ONGOING_PROGRESS_CUTOFF = 100  # projects below this are still "live" -> early-warning candidates


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

    def train_regressor_with_conformal(y, confidence=CONFIDENCE_LEVEL):
        """Split conformal prediction: train the point-predictor on a training
        slice only, then measure real error on a held-out calibration slice.
        The resulting interval width is backed by that measured error, not an
        assumption - valid even with a modest dataset, unlike quantile
        regression which needs more data per quantile to avoid crossing."""
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        m = xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=0.8, reg_lambda=2.0, random_state=42,
        )
        m.fit(X_train, y_train)

        calib_preds = m.predict(X_calib)
        residuals = np.abs(y_calib.values - calib_preds)
        n = len(residuals)
        # standard split-conformal quantile correction (Vovk et al.)
        q_level = min(1.0, np.ceil((n + 1) * confidence) / n)
        interval_halfwidth = float(np.quantile(residuals, q_level)) if n > 0 else float("nan")

        return m, interval_halfwidth, n

    cost_model, cost_cv = train_model(df["cost_at_risk"])
    time_model, time_cv = train_model(df["time_at_risk"])

    def train_baseline(y):
        # Plain logistic regression on identical features/folds - the honest
        # answer to "does the fancier model actually buy us anything over
        # conventional statistics?", which the PS explicitly asks teams to check.
        b = LogisticRegression(max_iter=1000)
        try:
            cv = cross_val_score(b, X, y, cv=5, scoring="roc_auc")
        except ValueError:
            cv = np.array([np.nan])
        b.fit(X, y)
        return b, cv

    _, cost_stat_cv = train_baseline(df["cost_at_risk"])
    _, time_stat_cv = train_baseline(df["time_at_risk"])

    cost_regressor, cost_interval_hw, cost_n_calib = train_regressor_with_conformal(df["cost_overrun_pct"])
    time_regressor, time_interval_hw, time_n_calib = train_regressor_with_conformal(df["time_overrun_months"])

    cost_explainer = shap.TreeExplainer(cost_model)
    time_explainer = shap.TreeExplainer(time_model)

    # Score every row once, at load time, so the Early Warning page doesn't
    # retrain or re-score per view - it just filters/sorts a precomputed column.
    df["cost_risk_score"] = cost_model.predict_proba(X)[:, 1]
    df["time_risk_score"] = time_model.predict_proba(X)[:, 1]
    df["overall_risk_score"] = df[["cost_risk_score", "time_risk_score"]].max(axis=1)

    return {
        "df": df, "X_cols": X, "rcf_baseline": rcf_baseline,
        "cost_model": cost_model, "cost_cv": cost_cv, "cost_explainer": cost_explainer,
        "time_model": time_model, "time_cv": time_cv, "time_explainer": time_explainer,
        "cost_regressor": cost_regressor, "cost_interval_hw": cost_interval_hw, "cost_n_calib": cost_n_calib,
        "time_regressor": time_regressor, "time_interval_hw": time_interval_hw, "time_n_calib": time_n_calib,
        "cost_stat_cv": cost_stat_cv, "time_stat_cv": time_stat_cv,
    }


def predict(state_dict, project_dict):
    """project_dict: {original_cost_cr, physical_progress_pct, ministry, state}
    Returns a dict with classification risk scores, SHAP factors, AND continuous
    expected-value predictions with conformal confidence intervals."""
    row = pd.DataFrame([project_dict])
    row_encoded = pd.get_dummies(row, columns=CAT_FEATURES).reindex(
        columns=state_dict["X_cols"].columns, fill_value=0
    )

    cost_score = float(state_dict["cost_model"].predict_proba(row_encoded)[0][1])
    time_score = float(state_dict["time_model"].predict_proba(row_encoded)[0][1])

    cost_expected = float(state_dict["cost_regressor"].predict(row_encoded)[0])
    time_expected = float(state_dict["time_regressor"].predict(row_encoded)[0])
    cost_hw = state_dict["cost_interval_hw"]
    time_hw = state_dict["time_interval_hw"]

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

    return {
        "cost_score": cost_score, "time_score": time_score,
        "cost_factors": grouped_shap(state_dict["cost_explainer"]),
        "time_factors": grouped_shap(state_dict["time_explainer"]),
        "cost_expected": cost_expected,
        "cost_low": cost_expected - cost_hw, "cost_high": cost_expected + cost_hw,
        "time_expected": time_expected,
        "time_low": time_expected - time_hw, "time_high": time_expected + time_hw,
    }
