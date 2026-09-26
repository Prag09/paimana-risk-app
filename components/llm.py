"""LLM layer for PAIMANA: plain-language explanations and Ask PAIMANA Q&A.

Provider order:
  1. Groq      (GROQ_API_KEY)       - primary, free tier, very fast
  2. Claude    (ANTHROPIC_API_KEY)  - optional fallback, only if a key is set
  3. Friendly fallback message      - never shows a raw error to the viewer

Key lookup order for each provider: environment variable -> st.secrets -> none.
Keys live in .streamlit/secrets.toml locally (gitignored) and in
Streamlit Cloud -> App settings -> Secrets for the live deployment.
"""

import os
import time
import streamlit as st

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Kept so existing imports in app.py don't break.
GEMINI_AVAILABLE = GROQ_AVAILABLE or ANTHROPIC_AVAILABLE

# Groq models (see console.groq.com/docs/models).
GROQ_MODEL = "llama-3.3-70b-versatile"        # main model: good quality, fast
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"  # used if the main model is busy/rate limited

# Claude models, used only if ANTHROPIC_API_KEY is set and Groq fails.
LLM_MODEL = "claude-sonnet-5"
LLM_MODEL_DEEP = "claude-opus-5-5"

REQUEST_TIMEOUT_S = 12
ONGOING_PROGRESS_CUTOFF = 100

FALLBACK_MESSAGE = (
    "The AI assistant is busy right now. The risk analysis on this page is "
    "unaffected - please try again in a few seconds."
)

SYSTEM_PROMPT_BASE = (
    "You are a project-monitoring analyst assistant for India's PAIMANA "
    "infrastructure project database (MoSPI). Answer using only the summary "
    "data provided below. Be concise (a few sentences unless asked for more), "
    "cite specific ministries/states/projects/numbers from the summary, and "
    "say plainly when something isn't covered by the summary rather than "
    "guessing. Do not invent numbers.\n\n"
)


def _read_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def get_groq_key():
    return _read_secret("GROQ_API_KEY")


def get_anthropic_key():
    return _read_secret("ANTHROPIC_API_KEY")


def get_active_key():
    """Returns whichever provider key is configured (Groq preferred).
    app.py uses this only to check whether *any* AI provider is set up."""
    return get_groq_key() or get_anthropic_key()


# app.py calls get_gemini_key() to check that a key exists - keep it working.
get_gemini_key = get_active_key


def build_context_summary(df):
    """Compact, aggregated dataset summary fed to the LLM instead of raw rows -
    keeps prompts small/fast and keeps every claim traceable to real numbers."""
    lines = [
        f"Dataset: {len(df)} project records across {df['report_month'].nunique()} "
        f"month(s), {df['ministry'].nunique()} ministries, {df['state'].nunique()} states/UTs.",
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


def _call_groq(prompt, system_prompt, model):
    client = groq.Groq(api_key=get_groq_key(), timeout=REQUEST_TIMEOUT_S, max_retries=1)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def _call_claude(prompt, system_prompt, model):
    client = anthropic.Anthropic(api_key=get_anthropic_key(), timeout=REQUEST_TIMEOUT_S)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def call_llm(prompt, context="", model=None):
    """Same signature as before, so app.py needs no changes.
    `model` is only used for the Claude fallback (e.g. LLM_MODEL_DEEP)."""
    system_prompt = SYSTEM_PROMPT_BASE + context

    if not get_active_key():
        return ("AI assistant not configured. Add `GROQ_API_KEY` to "
                "`.streamlit/secrets.toml` locally, and to Streamlit Cloud "
                "App settings -> Secrets for the live site.")

    # 1) Groq: main model, then the smaller model if the main one is busy.
    if GROQ_AVAILABLE and get_groq_key():
        for groq_model in (GROQ_MODEL, GROQ_MODEL_FALLBACK):
            try:
                answer = _call_groq(prompt, system_prompt, groq_model)
                if answer:
                    return answer
            except groq.AuthenticationError:
                return "AI assistant error: the Groq API key is invalid. Check GROQ_API_KEY in Secrets."
            except Exception:
                time.sleep(1)  # brief pause, then try the next model

    # 2) Claude, only if a key is configured.
    if ANTHROPIC_AVAILABLE and get_anthropic_key():
        try:
            answer = _call_claude(prompt, system_prompt, model or LLM_MODEL)
            if answer:
                return answer
        except Exception:
            pass

    # 3) Never show a stack trace to the judges.
    return FALLBACK_MESSAGE