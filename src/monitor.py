"""Score the model over time-ordered windows and log precision/recall per window.

Appends to data/metrics_history.csv, which the Streamlit dashboard plots to show
precision/recall drift. In production, each window would be a day or week of
newly labelled transactions.
"""
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.train import FEATURES, MODEL_DIR, load_data

HISTORY = Path("data/metrics_history.csv")


def evaluate_windows(df: pd.DataFrame, model, n_windows: int = 6) -> pd.DataFrame:
    df = df.sort_values("Time").reset_index(drop=True)
    rows = []
    size = len(df) // n_windows
    for i in range(n_windows):
        chunk = df.iloc[i * size:(i + 1) * size]
        if chunk["Class"].sum() == 0:
            continue
        pred = model.predict(chunk[FEATURES])
        rows.append({
            "window": i + 1,
            "precision": precision_score(chunk["Class"], pred, zero_division=0),
            "recall": recall_score(chunk["Class"], pred, zero_division=0),
            "fraud_in_window": int(chunk["Class"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    model = joblib.load(MODEL_DIR / "model.joblib")
    holdout = MODEL_DIR / "holdout.csv"
    # Score only rows the model never saw; in production use fresh labelled data
    data = pd.read_csv(holdout) if holdout.exists() else load_data()
    result = evaluate_windows(data, model, n_windows=4)
    result["logged_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    HISTORY.parent.mkdir(exist_ok=True)
    result.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)
    print(result)


if __name__ == "__main__":
    main()
