import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.eda import summarize
from src.generate_data import generate
from src.monitor import evaluate_windows
from src.train import FEATURES, train

PARAMS = {"n_estimators": 40, "max_depth": 4, "learning_rate": 0.2}


@pytest.fixture(scope="module")
def df():
    return generate(n_rows=12_000, seed=7)


@pytest.fixture(scope="module")
def trained(df):
    return train(df, PARAMS)


def test_data_schema_and_imbalance(df):
    assert list(df.columns) == FEATURES + ["Class"]
    stats = summarize(df)
    assert stats["missing_values"] == 0
    assert 0 < stats["fraud_rate"] < 0.01  # severe class imbalance


def test_model_catches_most_fraud(trained):
    _, metrics = trained
    assert metrics["recall"] > 0.6
    assert metrics["roc_auc"] > 0.9
    assert set(metrics) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_monitor_windows(df, trained):
    model, _ = trained
    out = evaluate_windows(df, model, n_windows=3)
    assert {"window", "precision", "recall"} <= set(out.columns)
    assert out["recall"].between(0, 1).all()


def test_api_health_and_predict(df, trained, monkeypatch):
    model, _ = trained
    monkeypatch.setattr(app_module, "_model", model)
    client = TestClient(app_module.app)

    assert client.get("/health").status_code == 200

    fraud_row = df[df.Class == 1].iloc[0][FEATURES].to_dict()
    resp = client.post("/predict", json=fraud_row)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_api_rejects_negative_amount(trained, monkeypatch):
    monkeypatch.setattr(app_module, "_model", trained[0])
    client = TestClient(app_module.app)
    assert client.post("/predict", json={"Amount": -5}).status_code == 422
