"""Train an XGBoost fraud classifier with SMOTE and MLflow tracking."""
import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.eda import summarize
from src.generate_data import generate

DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"{path} not found, using synthetic data (see src/generate_data.py)")
    return generate()


def split_data(df: pd.DataFrame, seed: int = 42):
    X, y = df[FEATURES], df["Class"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def train(df: pd.DataFrame, params: dict, seed: int = 42):
    X_tr, X_te, y_tr, y_te = split_data(df, seed)
    # SMOTE on the training split ONLY, so the test set stays realistic
    X_res, y_res = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)

    model = XGBClassifier(**params, random_state=seed, eval_metric="aucpr")
    model.fit(X_res, y_res)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fraud-detection")

    df = load_data()
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_params({f"data_{k}": v for k, v in summarize(df).items()})
        model, metrics = train(df, params)
        mlflow.log_metrics(metrics)

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        # Hold-out rows the model never trained on; the monitor scores these
        _, X_te, _, y_te = split_data(df)
        X_te.assign(Class=y_te).to_csv(MODEL_DIR / "holdout.csv", index=False)
        mlflow.log_artifact(str(MODEL_DIR / "model.joblib"))
        mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
