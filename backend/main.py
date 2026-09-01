import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from typing import List

# ------------------------------
# 1. Load Models & Helpers
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load the trained model
model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))

# Load the scaler
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Load feature names
with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
    feature_names = json.load(f)

# Load the SHAP explainer (if exists, else create it)
try:
    explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    print("✅ SHAP explainer loaded successfully!")
except:
    # If SHAP explainer is not found, we import shap and create a new one
    import shap
    explainer = shap.TreeExplainer(model)
    print("⚠️ SHAP explainer re-created from model.")

# Import our helper functions from the file we uploaded
# But since we need to import, we'll just define them here to avoid import issues
# Or we can import from prediction_functions.py
import sys
sys.path.append(os.path.dirname(__file__))
from prediction_functions import predict_with_explanation

# ------------------------------
# 2. FastAPI App Setup
# ------------------------------
app = FastAPI(
    title="AI Customer Retention Platform API",
    description="Predict churn, get explanations, and recommendations",
    version="1.0.0"
)

# Enable CORS so your React frontend (Vercel) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For now, allow all during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 3. Pydantic Schemas (Request/Response)
# ------------------------------
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    InternetService: str
    Contract: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "InternetService": "Fiber optic",
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1020.5
            }
        }

class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    risk_color: str
    reasons: List[str]
    recommendation: str

# ------------------------------
# 4. API Endpoints
# ------------------------------
@app.get("/")
def root():
    return {
        "message": "AI Customer Retention Platform API is live!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a single customer.
    Accepts customer details, runs through the ML pipeline,
    and returns probability + risk + reasons + recommendation.
    """
    try:
        # Convert input to DataFrame with proper feature order
        input_dict = customer.dict()
        input_df = pd.DataFrame([input_dict])
        
        # We need to one-hot encode this input just like we did in training
        # This is a simplified version - we should have a full preprocessing pipeline
        # For now, we'll use the prediction_functions.py which handles it
        result = predict_with_explanation(
            customer_data_df=input_df,
            model=model,
            scaler=scaler,
            explainer=explainer,
            feature_names=feature_names
        )
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
def batch_predict(customers: List[CustomerInput]):
    """
    Predict for multiple customers at once (optional, good for batch processing).
    """
    results = []
    for customer in customers:
        input_dict = customer.dict()
        input_df = pd.DataFrame([input_dict])
        result = predict_with_explanation(
            customer_data_df=input_df,
            model=model,
            scaler=scaler,
            explainer=explainer,
            feature_names=feature_names
        )
        results.append(result)
    return {"results": results}
