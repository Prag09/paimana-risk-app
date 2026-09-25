"""Claude-powered Q&A / plain-language explanation layer.

Key lookup order: environment variable -> .streamlit/secrets.toml -> none.
Deliberately NO visible sidebar input field (per product decision) - the key
lives in secrets.toml (which MUST be gitignored, see note in app.py) or an
environment variable set at deploy time.
"""

import os
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# GEMINI_AVAILABLE kept as an alias so any lingering imports don't break;
# this app now runs entirely on the Claude API.
GEMINI_AVAILABLE = ANTHROPIC_AVAILABLE

LLM_MODEL = "claude-sonnet-5"  # used for the quick inline "explain in plain language" buttons
LLM_MODEL_DEEP = "claude-opus-5-5"  # used for Ask PAIMANA, where deeper reasoning is worth the latency

ONGOING_PROGRESS_CUTOFF = 100


def get_anthropic_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


# Kept as an alias - app.py calls get_gemini_key() in a few places.
get_gemini_key = get_anthropic_key


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


def call_llm(prompt, context="", model=None):
    api_key = get_anthropic_key()
    if not ANTHROPIC_AVAILABLE:
        return "The `anthropic` package isn't installed. Add `anthropic` to requirements.txt and redeploy."
    if not api_key:
        return ("No Claude API key configured. Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` "
                 "(local + Streamlit Cloud 'Secrets' settings) or as an environment variable.")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        system_prompt = (
            "You are a project-monitoring analyst assistant for India's PAIMANA "
            "infrastructure project database (MoSPI). Answer using only the summary "
            "data provided below. Be concise (a few sentences unless asked for more), "
            "cite specific ministries/states/projects/numbers from the summary, and "
            "say plainly when something isn't covered by the summary rather than "
            "guessing.\n\n" + context
        )
        response = client.messages.create(
            model=model or LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
    except anthropic.AuthenticationError:
        return "Claude API call failed: invalid API key."
    except anthropic.RateLimitError:
        return "Claude API call failed: rate limited, please try again shortly."
    except anthropic.APIStatusError as e:
        return f"Claude API call failed: {e.message}"
    except Exception:
        return "The AI explanation is temporarily unavailable — the risk analysis above is unaffected."
