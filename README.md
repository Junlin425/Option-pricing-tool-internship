# Option Pricing Tool Internship Project

This repository contains a quantitative research project for European call option pricing. The deployed Streamlit application compares three pricing approaches:

1. LSTM volatility forecast combined with BSM pricing
2. Linear Regression CallPrice prediction
3. Black-Scholes-Merton baseline pricing

The application also includes a manual BSM calculator, sensitivity charts, live JPM/VIX/US Treasury data, model performance metrics and historical error references.

## Streamlit deployment

- Main file: `Week7/pricing_tool/app.py`
- Python version: `3.12`
- Dependency file: `Week7/pricing_tool/requirements.txt`

Add the following values in Streamlit Community Cloud under **Advanced settings > Secrets**:

```toml
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key"
FRED_API_KEY = "your_fred_key"
```

Do not commit `.env` or `.streamlit/secrets.toml`.

## Run locally

Install the dependencies from the repository root:

```powershell
python -m pip install -r Week7/pricing_tool/requirements.txt
```

Create a local `.env` file from `.env.example`, then start the application:

```powershell
python -m streamlit run Week7/pricing_tool/app.py
```

## Required deployment files

The live application uses:

- `Week7/pricing_tool/` for the interface and pricing services
- `Week7/model_artifacts/` for the trained Linear Regression and LSTM models
- `Week7/data/vix_feature_thresholds.json` for VIX feature engineering
- `Week7/results/task_1_3_model_metrics.csv` and `task_1_3_test_predictions.csv` for the performance dashboard
- `Week7/model_deployment/artifact_training.py` and `Week7/sensitivity_analysis/model_variant_training.py` for the saved model schema and pricing utilities

## Important limitation

The CallPrice target used in the research is generated from the BSM framework rather than observed option market quotes. The application is a research prototype and does not provide trading or investment advice.
