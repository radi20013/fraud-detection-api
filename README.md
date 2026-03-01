# Fraud Detection API – Machine Learning & Anomaly Detection

## Overview

This project demonstrates an **end-to-end fraud detection system** built using
**unsupervised machine learning** and deployed via a **FastAPI REST API**.

The workflow covers the full data science lifecycle:
exploratory analysis, preprocessing, model training, threshold calibration,
and real-time inference through an API.

Two anomaly detection approaches are implemented and compared:

- **Isolation Forest** (tree-based anomaly detection)
- **Autoencoder Neural Network** (reconstruction-error based detection)

The project focuses on *realistic fraud detection constraints*, including
high class imbalance and limited labels.

---

## Key Features

- Exploratory Data Analysis (EDA) of transaction data
- Data preprocessing and feature scaling pipeline
- Isolation Forest anomaly detection
- Deep Autoencoder anomaly detection (TensorFlow / Keras)
- Threshold calibration and model comparison
- FastAPI backend for real-time scoring
- REST endpoints for single and batch inference

---

## Repository Structure

fraud-detection-api/
│
├── Notebooks/ # Data analysis & model development
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_isolation_forest.ipynb
│ ├── 04_autoencoder.ipynb
│ └── 05_model_comparison.ipynb
│
├── fraud_api/ # Production API
│ ├── main.py # FastAPI application
│ ├── thresholds.json # Anomaly thresholds
│ └── autoencoder_model/ # Saved TensorFlow model
│
├── .gitignore
└── README.md


Large datasets are intentionally excluded from the repository and ignored via
`.gitignore` to follow best practices.

---

## Models Used

### Isolation Forest
- Unsupervised, tree-based anomaly detection
- Identifies rare and unusual transactions by isolation depth
- Produces an anomaly score for each transaction

### Autoencoder
- Neural network trained only on normal transactions
- Detects anomalies via reconstruction error
- Threshold selected from training error distribution

Both models are evaluated and compared to analyse agreement and behaviour.

---

## API Endpoints

The FastAPI service exposes the following endpoints:

- `GET /`  
  Health check

- `GET /example`  
  Returns a sample transaction payload

- `POST /score`  
  Scores a single transaction using both models

- `POST /compare`  
  Scores multiple transactions and reports model agreement

Interactive API documentation is available via Swagger UI.

---

## Running the API Locally

### Install dependencies
```bash
pip install fastapi uvicorn numpy pandas scikit-learn tensorflow joblib

Start the API
cd fraud_api
uvicorn main:app --reload

Open API docs
http://127.0.0.1:8000/docs

Example Response
{
  "iso_anomaly_score": -0.2487,
  "iso_flag": 0,
  "ae_reconstruction_error": 0.5606,
  "ae_flag": 0
}

What This Project Demonstrates

1)End-to-end machine learning workflow
2)Unsupervised anomaly detection for fraud
3)Handling highly imbalanced data
4)Model threshold calibration
5)Model comparison and agreement analysis
6)Deploying ML models via a REST API
6)Clean project structure and version control

Future Improvements

1)Dockerise the API for deployment
2)Cloud deployment (AWS / Azure / GCP)
3)Monitoring and model drift detection
4)Authentication and rate limiting

Author

Radi
Machine Learning & Data Science Portfolio Project