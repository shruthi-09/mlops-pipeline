import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the latest trained model
model = mlflow.sklearn.load_model("models/churn_model")

app = FastAPI(title="Churn Prediction API", version="1.0")

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "result": "Will Churn" if prediction == 1 else "Will Not Churn"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)