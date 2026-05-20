import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

def train_model():
    """Train churn prediction model with MLflow tracking."""
    print("Loading processed data...")
    df = pd.read_csv("data/processed/churn_processed.csv")

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        # Model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42
        }

        # Train model
        print("Training model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"AUC-ROC  : {auc:.4f}")
        print("Model logged to MLflow!")

if __name__ == "__main__":
    train_model()