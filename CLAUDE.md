# PAIMANA — Infrastructure Risk Intelligence

Streamlit dashboard for SIH PS SIH26103: predictive infrastructure
monitoring and early warning, built on real MoSPI Flash Report data.

## Tech stack

- **Frontend**: `streamlit`, `plotly`
- **ML**: `xgboost` (classification + regression), `scikit-learn`
  (cross-validation, Logistic Regression baseline), `shap`
  (explainability), custom split conformal prediction for confidence
  intervals
- **AI assistant**: Anthropic Claude API (`anthropic` SDK)
- **Data pipeline**: `pdftotext` + regex extraction (`ingest_report.py`)
  into `data/projects_master.csv`

## Running locally

```bash
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

To ingest a new month's Flash Report:
```bash
python ingest_report.py path/to/FlashReport.pdf YYYY-MM
```

## Rules

- Don't change model/ML code (`components/data_model.py`, XGBoost
  params, SHAP, conformal intervals) without asking first.
- Don't commit `.streamlit/secrets.toml` or `.env` — API keys only via
  `st.secrets` or environment variables.
- Keep files under 500 lines where reasonable.
- This is a live hackathon demo app — prefer small, verifiable changes
  and run `streamlit run app.py` after edits before committing.
