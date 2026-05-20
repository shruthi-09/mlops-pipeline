import pandas as pd

def load_churn_data():
    """Load and save raw churn dataset."""
    print("Downloading dataset...")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )
    df.to_csv("data/raw/churn.csv", index=False)
    print(f"Dataset saved! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_churn_data()
    print(df.head())
    print(df.dtypes)