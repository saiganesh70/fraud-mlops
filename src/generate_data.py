"""Synthetic stand-in for the Kaggle Credit Card Fraud dataset.

Same schema (Time, V1..V28, Amount, Class) and similar class imbalance (~0.17%
fraud), so the whole pipeline runs without downloading anything. For the real
project, put Kaggle's creditcard.csv at data/creditcard.csv instead.
"""
import numpy as np
import pandas as pd


def generate(n_rows: int = 60_000, fraud_rate: float = 0.0017, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 20)
    n_legit = n_rows - n_fraud

    legit = rng.normal(0, 1, size=(n_legit, 28))
    # Fraud differs in a handful of components, mimicking PCA-transformed signals
    shift = np.zeros(28)
    shift[[1, 3, 9, 10, 11, 13, 16]] = [2.5, -3.0, -2.5, 2.8, -3.2, -3.5, -2.0]
    fraud = rng.normal(0, 1.3, size=(n_fraud, 28)) + shift

    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    amount = np.where(
        y == 1, rng.lognormal(4.0, 1.4, len(y)), rng.lognormal(3.4, 1.2, len(y))
    ).round(2)
    time = rng.uniform(0, 172_800, len(y))  # two days in seconds

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = y
    return df.sort_values("Time").reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/creditcard.csv", index=False)
    print(f"Wrote data/creditcard.csv: {len(df)} rows, {df.Class.sum()} fraud")
