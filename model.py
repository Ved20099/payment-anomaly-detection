import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def load_and_train(filepath=None):
    import os
    if filepath is None:
        if os.path.exists("data/hotel_transactions.csv"):
            filepath = "data/hotel_transactions.csv"
        else:
            filepath = "hotel_transactions.csv"
    df = pd.read_csv(filepath)

    # ── Feature engineering ───────────────────────────────────
    # Amount per night ratio
    df["amount_per_night"] = df["amount"] / df["nights"].replace(0, 1)

    # Is it a refund
    df["is_refund"] = (df["amount"] < 0).astype(int)
    df["abs_amount"] = df["amount"].abs()

    # Is it a cash transaction
    df["is_cash"] = (df["payment_method"] == "Cash").astype(int)

    # Is it an odd hour (1am-4am)
    df["is_odd_hour"] = df["transaction_hour"].apply(
        lambda h: 1 if h <= 4 or h == 0 else 0
    )

    # Amount deviation from expected room rate
    df["rate_deviation"] = abs(df["amount_per_night"] - df["base_rate"])
    df["rate_deviation_pct"] = df["rate_deviation"] / df["base_rate"].replace(0, 1)

    # ── Select features for model ─────────────────────────────
    features = [
        "abs_amount",
        "amount_per_night",
        "transaction_hour",
        "nights",
        "base_rate",
        "rate_deviation",
        "rate_deviation_pct",
        "is_cash",
        "is_refund",
        "is_odd_hour",
    ]

    X = df[features].fillna(0)

    # ── Scale ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train Isolation Forest ────────────────────────────────
    model = IsolationForest(
        contamination=0.02,
        random_state=42,
        n_estimators=150,
        max_samples="auto",
    )
    model.fit(X_scaled)

    # ── Predictions ───────────────────────────────────────────
    df["anomaly_score"]    = model.decision_function(X_scaled)
    df["predicted_anomaly"] = model.predict(X_scaled)
    df["predicted_anomaly"] = df["predicted_anomaly"].apply(
        lambda x: 1 if x == -1 else 0
    )

    # Normalise score to 0-100 risk scale (higher = riskier)
    min_s = df["anomaly_score"].min()
    max_s = df["anomaly_score"].max()
    df["risk_score"] = ((df["anomaly_score"] - max_s) / (min_s - max_s) * 100).round(1)

    return df, model, scaler, features


if __name__ == "__main__":
    df, model, scaler, features = load_and_train()
    print(f"Predicted anomalies : {df['predicted_anomaly'].sum()}")
    print(f"Actual anomalies    : {df['is_anomaly'].sum()}")
    tp = ((df['predicted_anomaly'] == 1) & (df['is_anomaly'] == 1)).sum()
    print(f"True positives      : {tp}")
