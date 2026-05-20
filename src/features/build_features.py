import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def build_features(input_path="data/raw/churn.csv", 
                   output_path="data/processed/churn_processed.csv"):
    """Clean and engineer features from raw churn data."""
    print("Building features...")
    df = pd.read_csv(input_path)

    # Drop customerID - not useful for prediction
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges - it has spaces, convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode all categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features saved! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = build_features()
    print(df.head())