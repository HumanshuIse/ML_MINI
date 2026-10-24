from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data_loader import discover_csvs, load_market_data
from src.feature_engineering import build_features
from src.market_analysis import annual_summary, metrics, top_movements
from src.model_training import train_models
from src.preprocessing import filter_dates, prepare_series
from src.report_generator import make_report

st.set_page_config(page_title="MarketPulse India", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp {background: #071525; color: #e8f1f5}
[data-testid="stMetric"] {background:#10263a; border:1px solid #1d4960; padding:15px; border-radius:14px}
.block-container {max-width:1400px; padding-top:2rem}
h1,h2,h3 {color:#7ee7e0}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_load(path: str, modified: float):
    return load_market_data(path)


def get_data():
    uploaded = st.sidebar.file_uploader("Upload Kaggle CSV", type=["csv"])
    if uploaded:
        return load_market_data(uploaded)
    files = discover_csvs(ROOT / "data")
    if files:
        return cached_load(str(files[0]), files[0].stat().st_mtime)
    return None, None


def chart_download(data: pd.DataFrame, label: str, filename: str):
    st.download_button(f"Download {label}", data.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def render_dashboard(data, quality, selected_index, date_start, date_end):
    frame = prepare_series(filter_dates(data, date_start, date_end), selected_index)
    if frame.empty:
        st.warning("No observations match the selected filters.")
        return
    summary = metrics(frame)
    st.title("MarketPulse India — Market Intelligence Dashboard")
    st.caption("Explore the past. Understand the market. Predict the next move.")
    cards = st.columns(5)
    cards[0].metric("Selected index", selected_index)
    cards[1].metric("Latest close", f"{summary['latest_close']:,.2f}")
    cards[2].metric("Total return", f"{summary['total_return']:.2%}")
    cards[3].metric("Annualized volatility", f"{summary['annualized_volatility']:.2%}")
    cards[4].metric("Maximum drawdown", f"{summary['max_drawdown']:.2%}")
    st.plotly_chart(px.line(frame, x="date", y="close", title="Historical closing price"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(frame, x="date", y="cumulative_return", title="Cumulative return"), use_container_width=True)
    with right:
        st.plotly_chart(px.line(frame, x="date", y="rolling_volatility", title="20-observation annualized volatility"), use_container_width=True)
    st.plotly_chart(px.bar(annual_summary(frame), x="year", y="annual_return", title="Annual performance"), use_container_width=True)
    st.info("Index level is a price-like measure. Return measures percentage change, making periods and indices easier to compare.")
    chart_download(frame, "cleaned analysis", "marketpulse_cleaned.csv")


def render_explorer(data, selected_index, date_start, date_end):
    st.header("Market Explorer")
    frame = prepare_series(filter_dates(data, date_start, date_end), selected_index)
    st.plotly_chart(px.line(frame, x="date", y=["open", "high", "low", "close"], title="OHLC history") if all(c in frame for c in ["open", "high", "low"]) else px.line(frame, x="date", y="close", title="Closing history"), use_container_width=True)
    down, up = top_movements(frame)
    a, b = st.columns(2)
    with a:
        st.subheader("Largest declines"); st.dataframe(down[["date", "daily_return", "close"]], hide_index=True)
    with b:
        st.subheader("Largest gains"); st.dataframe(up[["date", "daily_return", "close"]], hide_index=True)
    st.subheader("Historical observations")
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_returns(data, selected_index, date_start, date_end):
    st.header("Returns and Volatility")
    frame = prepare_series(filter_dates(data, date_start, date_end), selected_index)
    threshold = frame["rolling_volatility"].quantile(.9)
    st.caption(f"High-volatility periods are flagged using the 90th percentile of the rolling 20-observation annualized volatility ({threshold:.2%}).")
    a, b = st.columns(2)
    with a: st.plotly_chart(px.line(frame, x="date", y="daily_return", title="Daily simple return"), use_container_width=True)
    with b: st.plotly_chart(px.histogram(frame, x="daily_return", title="Daily return distribution"), use_container_width=True)
    st.plotly_chart(px.line(frame, x="date", y=["rolling_volatility", "drawdown"], title="Volatility and drawdown"), use_container_width=True)
    st.dataframe(annual_summary(frame), hide_index=True)
    chart_download(frame[["date", "close", "daily_return", "log_return", "cumulative_return", "rolling_volatility", "drawdown"]], "returns", "marketpulse_returns.csv")


def render_ml(data, selected_index, date_start, date_end):
    st.header("ML Prediction Lab & Model Arena")
    raw = filter_dates(data, date_start, date_end)
    series = raw[raw["index"] == selected_index].copy()
    features_frame, features = build_features(series)
    st.write("Features use only current/past data:", ", ".join(features))
    st.caption("Chronological split: earliest 70% train, next 15% validation, latest 15% test. The validation partition is reserved for future model selection.")
    selected = st.multiselect("Models", ["Zero-Return Baseline", "Linear Regression", "Random Forest", "Gradient Boosting"], default=["Zero-Return Baseline", "Linear Regression", "Random Forest", "Gradient Boosting"])
    if len(features_frame) < 15:
        st.warning("Not enough usable observations after feature construction.")
        return
    if st.button("Train selected models", type="primary"):
        with st.spinner("Training chronological models..."):
            st.session_state["results"] = train_models(features_frame, features, selected)
    results = st.session_state.get("results")
    if not results:
        st.info("Train models to view actual held-out results and a next-observation prediction.")
        return
    table = pd.DataFrame({name: values["metrics"] for name, values in results.items()}).T.reset_index(names="model")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.plotly_chart(px.bar(table, x="model", y="rmse", title="Test RMSE (lower is better)"), use_container_width=True)
    model_name = st.selectbox("Selected model for next-observation forecast", list(results))
    chosen = results[model_name]
    pred = chosen["predicted"]
    actual = chosen["actual"]
    pred_frame = pd.DataFrame({"date": chosen["test_dates"], "actual_return": actual, "predicted_return": pred})
    st.plotly_chart(px.line(pred_frame, x="date", y=["actual_return", "predicted_return"], title="Held-out actual vs predicted returns"), use_container_width=True)
    latest = features_frame.iloc[[-1]]
    model = chosen["model"]
    next_return = 0.0 if model is None else float(model.predict(latest[features])[0])
    latest_close = float(latest["close"].iloc[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest close", f"{latest_close:,.2f}")
    c2.metric("Predicted next return", f"{next_return:.3%}")
    c3.metric("Indicative next level", f"{latest_close * (1 + next_return):,.2f}")
    st.warning("This is an educational next-observation transformation, not a guaranteed price or financial advice.")
    chart_download(pred_frame, "test predictions", "marketpulse_predictions.csv")
    html = make_report(quality if (quality := st.session_state.get("quality")) else type("Q", (), {"warnings": [], "source": "loaded CSV", "rows_cleaned": len(data)})(), selected_index, date_start, date_end, results)
    st.download_button("Download HTML report", html, "marketpulse_report.html", "text/html")


def main():
    st.sidebar.title("📈 MarketPulse India")
    st.sidebar.caption("Educational market analytics and ML prototype")
    try:
        data, quality = get_data()
    except Exception as exc:
        st.error(f"Could not load the CSV: {exc}")
        st.stop()
    if data is None:
        st.title("MarketPulse India")
        st.info("Upload the Kaggle CSV in the sidebar, or place a CSV in data/. No fabricated charts or predictions are shown.")
        st.markdown("### Quick start\n1. Download the dataset from Kaggle.\n2. Place the CSV in `data/` or upload it.\n3. Choose a page and explore the actual observations.")
        return
    st.session_state["quality"] = quality
    indices = sorted(data["index"].unique().tolist())
    selected_index = st.sidebar.selectbox("Index", indices)
    min_date, max_date = data["date"].min().date(), data["date"].max().date()
    date_range = st.sidebar.slider("Analysis date range", min_date, max_date, (max(min_date, pd.Timestamp("2000-01-01").date()), min(max_date, pd.Timestamp("2020-12-31").date())))
    st.sidebar.caption(f"Coverage: {quality.coverage_start} to {quality.coverage_end} | {quality.rows_cleaned:,} rows")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Market Explorer", "Returns and Volatility", "ML Prediction Lab", "Data Quality", "Project Information"])
    if page == "Dashboard": render_dashboard(data, quality, selected_index, *date_range)
    elif page == "Market Explorer": render_explorer(data, selected_index, *date_range)
    elif page == "Returns and Volatility": render_returns(data, selected_index, *date_range)
    elif page == "ML Prediction Lab": render_ml(data, selected_index, *date_range)
    elif page == "Data Quality":
        st.header("Data Quality"); st.json(quality.__dict__); st.dataframe(data.head(20), hide_index=True)
    else:
        st.header("Project Information")
        st.markdown("**Dataset:** Kaggle Stock Market Sensex/Nifty All-Time Dataset  \n**Study period:** 2000–2020  \n**Target:** next available trading observation's close return  \n**Models:** zero-return baseline, linear regression, random forest, gradient boosting")
        st.info("Chronological splitting prevents future information leaking into training. Daily return prediction is difficult; complex models are not automatically better than the baseline.")


if __name__ == "__main__":
    main()

