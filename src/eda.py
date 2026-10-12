"""Quick EDA summary: class imbalance, amount stats, missing values."""
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    fraud = df["Class"].sum()
    return {
        "rows": int(len(df)),
        "fraud_count": int(fraud),
        "fraud_rate": float(fraud / len(df)),
        "missing_values": int(df.isna().sum().sum()),
        "amount_mean_legit": float(df.loc[df.Class == 0, "Amount"].mean()),
        "amount_mean_fraud": float(df.loc[df.Class == 1, "Amount"].mean()),
    }


if __name__ == "__main__":
    import json
    from src.train import load_data
    print(json.dumps(summarize(load_data()), indent=2))
