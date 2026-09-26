# PAIMANA — Infrastructure Risk Intelligence

**AI-assisted early warning for cost and schedule overruns in India's infrastructure projects.**

Built for **SIH26103** · Smart India Hackathon 2026 · Ministry of Statistics and Programme Implementation (MoSPI) · Team Void Pointers

🔗 **Live app**: [paimana-risk-null-pointers.streamlit.app](https://paimana-risk-null-pointers.streamlit.app)

---

## The problem

MoSPI's [PAIMANA portal](https://paimana-proj.mospi.gov.in) tracks India's large infrastructure projects (₹150+ Cr) — but only reports what has *already* happened. Cost and schedule overruns get documented after the fact, not flagged before they become unavoidable.

**PAIMANA (this project)** turns 20+ years of historical project patterns into an early-warning system: predict which ongoing projects are at risk of cost or schedule overrun, explain *why* in plain language, and surface it in a dashboard a policymaker can actually use.

## What it does

| Page | Answers |
|---|---|
| **Overview** | What's happening across the whole portfolio right now? |
| **Risk Assessment** | How risky is this specific project? (cost risk, schedule risk, expected overrun with confidence intervals) |
| **Early Warning** | Which *real, currently ongoing* projects should be flagged today? |
| **Ask PAIMANA** | Natural-language Q&A grounded in the real dataset (Claude-powered) |
| **Search Projects** | Find any project by name, region, ministry, cost, or reporting month |
| **Regional Intelligence** | Where is risk geographically concentrated? |
| **Trends** | How does risk relate to progress, ministry, and season? |
| **Add Project** | Add a project and see it scored live in the current session |
| **Methodology** | How does the model work, and what are its limits? |

## How it works

```
MoSPI Flash Report PDFs (monthly)
        │
        ▼
ingest_report.py  →  regex-based PDF parser + data-quality gate
        │
        ▼
data/projects_master.csv  →  871 real, verified project records (Apr–Jul 2026)
        │
        ▼
components/data_model.py
  ├─ XGBoost classifiers (cost-risk, schedule-risk)
  ├─ XGBoost regressors + split conformal prediction intervals
  ├─ SHAP explainability
  └─ Logistic Regression baseline (validates ML actually helps)
        │
        ▼
app.py + components/{styles,cards,charts,navigation,llm}.py
  → Streamlit dashboard, 9 pages
        │
        ▼
GitHub → Streamlit Community Cloud (auto-redeploys on push)
```

## Tech stack

- **Data**: `pdftotext`, regex-based extraction, `pandas`
- **ML**: `xgboost` (classification + regression), `scikit-learn` (cross-validation, Logistic Regression baseline), `shap` (explainability), custom split conformal prediction for confidence intervals
- **AI assistant**: Anthropic Claude API (`anthropic` SDK)
- **Frontend**: `streamlit`, `plotly`
- **Deployment**: GitHub → Streamlit Community Cloud

## Key design decisions

- **Real data only** — every row is extracted from an actual MoSPI Flash Report PDF, not synthetic or third-party data
- **Two-tier risk output** — a risk *probability* (classifier) alongside a continuous *expected overrun* with a statistically valid confidence interval (conformal prediction), instead of collapsing everything into a single threshold
- **Explainability is not optional** — every prediction ships with SHAP-based reasoning, not just a number
- **Honesty over polish** — the Methodology page states validated performance numbers (5-fold CV, not cherry-picked), compares against a plain statistical baseline, and documents known limitations rather than hiding them

## Known limitations (documented, not hidden)

- Currently uses only CUF (official form) features — the CUF vs. non-CUF comparison the problem statement asks for is the next milestone
- Physical progress correlates with predicted risk partly due to reporting lag (a project only shows a "revised" date/cost once someone records a revision) — full survival analysis is the planned fix
- Confidence intervals are wide, reflecting a genuinely modest calibration sample — by design, not a bug
- `Add Project` entries are session-only (no persistent database on the free hosting tier)

## Running locally

```bash
git clone https://github.com/Prag09/paimana-risk-app.git
cd paimana-risk-app
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

To add a new month's data:
```bash
python ingest_report.py path/to/FlashReport.pdf YYYY-MM
```

## Project structure

```
paimana-risk-app/
├── app.py                    # Streamlit UI - page orchestration
├── ingest_report.py          # PDF → structured CSV pipeline
├── requirements.txt
├── .streamlit/
│   └── config.toml           # Theme
├── components/
│   ├── data_model.py         # All ML models + predict()
│   ├── llm.py                # Claude Q&A / plain-language explanations
│   ├── styles.py             # Palette + global CSS
│   ├── cards.py               # Reusable UI components
│   ├── charts.py              # Themed Plotly builders
│   └── navigation.py          # Sidebar page switcher
└── data/
    └── projects_master.csv   # Accumulated real dataset
```

## Team

**Team Void Pointers** — SIH26103, Smart India Hackathon 2026
