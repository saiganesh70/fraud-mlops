   **Live demo:** https://fraud-mlops-cgto.onrender.com/docs
# Fraud Detection with MLOps

XGBoost credit-card fraud detector with experiment tracking, a REST API, Docker,
a CI/CD pipeline and a precision/recall drift dashboard.

**Stack:** XGBoost, SMOTE (imbalanced-learn), MLflow, FastAPI, Docker, GitHub Actions, Streamlit

## Pipeline

```
creditcard.csv -> EDA -> train/test split -> SMOTE (train only) -> XGBoost
      -> MLflow (params, metrics, artifacts) -> model.joblib
      -> FastAPI /predict -> Docker image -> Docker Hub
      -> weekly GitHub Actions retrain + monitor -> metrics_history.csv -> Streamlit drift dashboard
```

## Quick start

```bash
pip install -r requirements.txt

# Option A: real data. Download Kaggle "Credit Card Fraud Detection" and save as
#           data/creditcard.csv
# Option B: do nothing. A synthetic dataset with the same schema is used.

python -m src.eda            # class imbalance summary
python -m src.train          # trains, logs to MLflow (sqlite:///mlflow.db)
python -m src.monitor        # logs precision/recall per time window
pytest                       # tests

uvicorn src.app:app --reload                  # API on :8000, docs at /docs
streamlit run dashboard/app.py                # drift dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare experiments
```

Predict:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Amount": 120.5, "V4": 2.1, "V11": -3.0}'
```

## Docker

```bash
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

The image trains a model at build time, so it ships ready to serve.

## CI/CD (GitHub Actions)

- `ci.yml`: on every push/PR runs the tests, builds the Docker image, smoke-tests
  `/health`, and on `main` pushes the image to Docker Hub.
- `retrain.yml`: every Monday (or manually) runs tests, retrains, logs windowed
  precision/recall, and commits `data/metrics_history.csv` for the dashboard.

Add repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` before the push step works.

## Design notes

- **SMOTE only on the training split.** Oversampling before splitting leaks
  synthetic copies into the test set and inflates metrics.
- **Precision/recall/PR-AUC, not accuracy.** With ~0.17% fraud, accuracy is meaningless.
- **Monitoring scores only held-out rows.** In production, replace them with newly
  labelled transactions to see real drift.
- The synthetic fallback data is much easier than the real dataset, so its scores
  (precision/recall above 0.9) are not representative. Report numbers from the real data.

## Results (Kaggle Credit Card Fraud dataset)

Trained locally on the full dataset (80/20 stratified split, SMOTE on train only, threshold 0.5):

| Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0.63 | 0.86 | 0.72 | 0.983 | 0.872 |

The live demo is trained on synthetic data (the real dataset is too large for the free hosting tier), so its scores are not representative.
