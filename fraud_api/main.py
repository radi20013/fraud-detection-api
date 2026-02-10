import json
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Fraud Anomaly Scoring API")

# Load saved assets
scaler = joblib.load("scaler.pkl")
print("Scaler fitted on:", scaler.feature_names_in_)  # Debug info

iso_model = joblib.load("isolation_forest.pkl")
ae_model = tf.keras.models.load_model("autoencoder_model")

with open("thresholds.json", "r") as f:
    thresholds = json.load(f)

ISO_T = float(thresholds["iso_threshold"])
AE_T = float(thresholds["ae_threshold"])

# Must match training feature order
FEATURE_ORDER = [
    "Time",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount"
]

# For demo input endpoint
SAMPLE_CSV = "X_test.csv"

@app.get("/")
def home():
    return {"status": "ok", "message": "Fraud API running"}

# ✅ Step 1: Example endpoint
@app.get("/example")
def example():
    if not os.path.exists(SAMPLE_CSV):
        raise HTTPException(
            status_code=404,
            detail=f"{SAMPLE_CSV} not found. Copy X_test.csv into fraud_api/ to enable /example."
        )

    df_ex = pd.read_csv(SAMPLE_CSV)
    row = df_ex.sample(1).iloc[0].to_dict()
    payload = {k: float(row[k]) for k in FEATURE_ORDER}
    return payload

# Original scoring endpoint
@app.post("/score")
def score(data: dict):
    missing = [k for k in FEATURE_ORDER if k not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing keys: {missing}")

    df = pd.DataFrame([[data[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER).astype(float)

    # IMPORTANT: scaler was fit on ["Amount", "Time"] in that order
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    X = df.values.astype("float32")

    iso_normality = iso_model.decision_function(X)[0]
    iso_score = float(-iso_normality)
    iso_flag = int(iso_score >= ISO_T)

    recon = ae_model.predict(X, verbose=0)
    ae_score = float(np.mean(np.square(X - recon)))
    ae_flag = int(ae_score >= AE_T)

    return {
        "iso_anomaly_score": iso_score,
        "iso_flag": iso_flag,
        "ae_reconstruction_error": ae_score,
        "ae_flag": ae_flag
    }

# ✅ Step 2: Compare endpoint
@app.post("/compare")
def compare(payloads: list[dict]):
    if len(payloads) == 0:
        raise HTTPException(status_code=400, detail="payloads list is empty")

    rows = []
    for i, data in enumerate(payloads):
        missing = [k for k in FEATURE_ORDER if k not in data]
        if missing:
            raise HTTPException(status_code=400, detail=f"Item {i} missing keys: {missing}")
        rows.append([data[k] for k in FEATURE_ORDER])

    df = pd.DataFrame(rows, columns=FEATURE_ORDER).astype(float)

    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])
    X = df.values.astype("float32")

    iso_normality = iso_model.decision_function(X)
    iso_scores = (-iso_normality).astype(float)
    iso_flags = (iso_scores >= ISO_T).astype(int)

    recon = ae_model.predict(X, verbose=0)
    ae_scores = np.mean(np.square(X - recon), axis=1).astype(float)
    ae_flags = (ae_scores >= AE_T).astype(int)

    agree = (iso_flags == ae_flags).astype(int)

    return {
        "n_samples": int(len(payloads)),
        "iso": {
            "threshold": ISO_T,
            "flagged": int(iso_flags.sum()),
            "avg_score": float(iso_scores.mean()),
            "max_score": float(iso_scores.max())
        },
        "ae": {
            "threshold": AE_T,
            "flagged": int(ae_flags.sum()),
            "avg_score": float(ae_scores.mean()),
            "max_score": float(ae_scores.max())
        },
        "agreement": {
            "same_flag_count": int(agree.sum()),
            "same_flag_rate": float(agree.mean())
        }
    }

