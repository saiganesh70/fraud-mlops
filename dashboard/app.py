"""Streamlit dashboard: precision/recall drift across monitoring runs."""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Model Monitor", layout="wide")
st.title("Fraud Model Monitoring")

try:
    df = pd.read_csv("data/metrics_history.csv")
except FileNotFoundError:
    st.warning("No history yet. Run: python -m src.monitor")
    st.stop()

latest = df[df["logged_at"] == df["logged_at"].max()]
c1, c2, c3 = st.columns(3)
c1.metric("Latest avg precision", f"{latest.precision.mean():.3f}")
c2.metric("Latest avg recall", f"{latest.recall.mean():.3f}")
c3.metric("Monitoring runs", df["logged_at"].nunique())

st.subheader("Precision and recall by time window (latest run)")
st.line_chart(latest.set_index("window")[["precision", "recall"]])

st.subheader("Average precision and recall across runs (drift over time)")
runs = df.groupby("logged_at")[["precision", "recall"]].mean()
st.line_chart(runs)
st.dataframe(df, use_container_width=True)
