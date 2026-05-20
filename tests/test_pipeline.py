import pytest
import pandas as pd
import numpy as np
from src.features.build_features import build_features
from src.data.load_data import load_churn_data

def test_data_shape():
    """Test that raw data loads with correct shape."""
    df = pd.read_csv("data/raw/churn.csv")
    assert df.shape[0] > 0, "Dataset should not be empty"
    assert df.shape[1] > 0, "Dataset should have columns"

def test_processed_data_exists():
    """Test that processed data file exists."""
    import os
    assert os.path.exists("data/processed/churn_processed.csv"), \
        "Processed data file should exist"

def test_no_missing_values():
    """Test that processed data has no missing values."""
    df = pd.read_csv("data/processed/churn_processed.csv")
    assert df.isnull().sum().sum() == 0, "Processed data should have no missing values"

def test_target_column_exists():
    """Test that Churn column exists in processed data."""
    df = pd.read_csv("data/processed/churn_processed.csv")
    assert "Churn" in df.columns, "Churn column should exist"

def test_target_is_binary():
    """Test that Churn column is binary (0 or 1)."""
    df = pd.read_csv("data/processed/churn_processed.csv")
    unique_values = df["Churn"].unique()
    assert set(unique_values).issubset({0, 1}), "Churn should be binary"

def test_model_exists():
    """Test that trained model exists."""
    import os
    assert os.path.exists("models/churn_model"), "Model should be saved"

def test_api_health():
    """Test that API health endpoint works."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_prediction():
    """Test that API prediction endpoint returns correct format."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    payload = {
        "gender": 1, "SeniorCitizen": 0, "Partner": 1,
        "Dependents": 0, "tenure": 12, "PhoneService": 1,
        "MultipleLines": 0, "InternetService": 1, "OnlineSecurity": 0,
        "OnlineBackup": 1, "DeviceProtection": 0, "TechSupport": 0,
        "StreamingTV": 1, "StreamingMovies": 0, "Contract": 0,
        "PaperlessBilling": 1, "PaymentMethod": 2,
        "MonthlyCharges": 65.5, "TotalCharges": 786.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
    assert "result" in response.json()