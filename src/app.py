"""FastAPI service that serves the trained fraud model."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.train import FEATURES, MODEL_DIR

app = FastAPI(title="Fraud Detection API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        path = Path(MODEL_DIR) / "model.joblib"
        if not path.exists():
            raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
        _model = joblib.load(path)
    return _model


class Transaction(BaseModel):
    Time: float = 0.0
    Amount: float = Field(default=0.0, ge=0)
    V1: float = 0.0; V2: float = 0.0; V3: float = 0.0; V4: float = 0.0
    V5: float = 0.0; V6: float = 0.0; V7: float = 0.0; V8: float = 0.0
    V9: float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (Path(MODEL_DIR) / "model.joblib").exists()}


@app.post("/predict")
def predict(tx: Transaction):
    model = get_model()
    row = pd.DataFrame([tx.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    return {"fraud_probability": round(prob, 4), "is_fraud": prob >= 0.5}
