# MarketPulse India

MarketPulse India is a Streamlit college mini project for analysing Sensex/Nifty history and predicting the **next available trading observation's return** with machine learning. It uses real CSV observations and never fabricates charts, metrics, or predictions when data is missing.

## Features

- Robust loading of common Kaggle CSV column names, including combined or single-index files.
- Date cleaning, numeric conversion, duplicate handling, OHLC diagnostics, and actual coverage reporting.
- Daily/log/cumulative returns, annual performance, rolling annualized volatility, drawdowns, movement tables, and interactive Plotly charts.
- Leakage-safe lag, momentum, range, volume, and rolling features.
- Chronological 70%/15%/15% train/validation/test split.
- Zero-return baseline, linear regression, random forest, and histogram gradient boosting.
- MAE, RMSE, R², directional accuracy, correlation, held-out actual-vs-predicted charts, next-observation forecast, CSV downloads, and HTML report.
- Data Quality and Project Information pages suitable for a classroom demonstration.

## Dataset

Download the [Kaggle dataset](https://www.kaggle.com/datasets/rockyt07/stock-marketsensex-nifty-all-time-dataset), place its CSV in `data/`, or upload it from the sidebar. The application analyses 1 January 2000 through 31 December 2020 by default and reports the actual coverage available.

Required fields are a date and close/price column. Open, high, low, volume, and index/symbol fields are optional. Index names are inferred from the file where possible.

## Installation on Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If the CSV is not present, the app opens in a data-needed state. This is intentional: it does not claim to have loaded the Kaggle data.

## Tests

```powershell
pytest -v
```

The tests use small synthetic fixtures only for unit behavior such as target alignment and leakage checks; the application requires the real CSV for analysis.

## Demonstration workflow

1. Load the CSV from `data/` or upload it.
2. Select Sensex or Nifty and confirm the displayed coverage.
3. Show the Dashboard and explain the difference between index level, return, volatility, and drawdown.
4. Use Market Explorer and Returns and Volatility to inspect best/worst days and annual behavior.
5. Open ML Prediction Lab, train all models, compare them with the zero-return baseline, and explain the chronological split.
6. Show the next-observation return and indicative level as an educational model transformation, not a guaranteed price.
7. Download the cleaned data, test predictions, or HTML report.

## Methodology and limitations

The target at date *t* is `(close[t+1] / close[t]) - 1`; features are built from date *t* and earlier observations. Rolling volatility assumes 252 trading observations per year. No random split or backward fill is used. R² can be uninformative for near-zero daily returns, so it is reported alongside MAE, RMSE, directional accuracy, and a zero-return baseline.

The project is a historical educational prototype, not financial advice. Historical relationships do not guarantee future results; daily index returns are noisy, and the indicative next level is not a promise. The dataset may contain source-specific OHLC anomalies, which are reported rather than silently rewritten.

## Structure

`app.py` contains the Streamlit UI; reusable loading, preprocessing, analysis, feature, model, evaluation, and report code lives in `src/`. Tests live in `tests/`. The full dataset and generated model artifacts are excluded from version control.
