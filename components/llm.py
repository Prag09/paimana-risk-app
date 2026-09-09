"""Gemini-powered Q&A / plain-language explanation layer.

Key lookup order: environment variable -> .streamlit/secrets.toml -> none.
Deliberately NO visible sidebar input field (per product decision) - the key
lives in secrets.toml (which MUST be gitignored, see note in app.py) or an
environment variable set at deploy time.
"""

import os
import streamlit as st

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

LLM_MODEL = "gemini-3.8-flash"  # verify current model names at ai.google.dev/gemini-api/docs/models
ONGOING_PROGRESS_CUTOFF = 100


def get_gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def build_context_summary(df):
    """Compact, aggregated dataset summary fed to the LLM instead of raw rows -
    keeps prompts small/fast and keeps every claim traceable to real numbers."""
    lines = [
        f"Dataset: {len(df)} project records across {df['report_month'].nunique()} "
        f"month(s), {df['ministry'].nunique()} ministries, {df['state'].nunique()} states.",
        "",
        "Ministry-level averages (avg cost overrun %, avg schedule slip months, project count):",
    ]
    by_ministry = (
        df.groupby("ministry")
        .agg(projects=("project_name", "count"),
             avg_cost_overrun_pct=("cost_overrun_pct", "mean"),
             avg_time_overrun_months=("time_overrun_months", "mean"))
        .round(1).sort_values("avg_cost_overrun_pct", ascending=False)
    )
    for ministry, row in by_ministry.iterrows():
        lines.append(f"- {ministry}: {int(row['projects'])} projects, "
                      f"{row['avg_cost_overrun_pct']}% avg cost overrun, "
                      f"{row['avg_time_overrun_months']} mo avg slip")

    lines.append("")
    lines.append("Top 10 ONGOING projects currently flagged highest overall risk:")
    top_risk = (
        df[df["physical_progress_pct"] < ONGOING_PROGRESS_CUTOFF]
        .sort_values("overall_risk_score", ascending=False).head(10)
    )
    for _, r in top_risk.iterrows():
        lines.append(f"- {r['project_name']} ({r['ministry']}, {r['state']}): "
                      f"{r['overall_risk_score']*100:.0f}% risk score, "
                      f"{r['physical_progress_pct']:.0f}% complete")
    return "\n".join(lines)


def call_llm(prompt, context=""):
    api_key = get_gemini_key()
    if not GEMINI_AVAILABLE:
        return "The `google-genai` package isn't installed. Add `google-genai` to requirements.txt and redeploy."
    if not api_key:
        return ("No Gemini API key configured. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` "
                 "(local + Streamlit Cloud 'Secrets' settings) or as an environment variable.")
    try:
        client = genai.Client(api_key=api_key)
        system_prompt = (
            "You are a project-monitoring analyst assistant for India's PAIMANA "
            "infrastructure project database (MoSPI). Answer using only the summary "
            "data provided below. Be concise (a few sentences unless asked for more), "
            "cite specific ministries/states/projects/numbers from the summary, and "
            "say plainly when something isn't covered by the summary rather than "
            "guessing.\n\n" + context
        )
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text
    except Exception as e:
        return f"LLM call failed: {e}"
