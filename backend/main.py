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

# Load feature names (the columns the model expects)
with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
    expected_features = json.load(f)

# Load SHAP explainer
try:
    import shap
    explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
except:
    import shap
    explainer = shap.TreeExplainer(model)
    print("⚠️ SHAP explainer re-created from model.")

# ------------------------------
# 2. PREPROCESSING FUNCTION (MOST IMPORTANT FIX)
# ------------------------------
def preprocess_customer(input_dict):
    """
    Convert raw customer JSON into the exact format the model expects.
    This mimics the one-hot encoding we did in Colab.
    """
    # Create a DataFrame with a single row
    df = pd.DataFrame([input_dict])
    
    # 1. Convert TotalCharges to numeric (handle empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # 2. Identify categorical columns (same as we did in training)
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # 3. One-hot encode ONLY the categorical columns that exist in the input
    # For columns not in input, we'll handle them later
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Ensure ALL expected columns are present (fill missing with 0)
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # 5. Reorder columns to match the training data exactly
    df_encoded = df_encoded[expected_features]
    
    return df_encoded

# ------------------------------
# 3. Helper Functions (Risk, SHAP, Recommendations)
# ------------------------------
def get_risk_category(probability):
    if probability < 0.30:
        return {'label': 'Low Risk', 'color': '🟢', 'bg_color': 'bg-green-500', 'action': 'No immediate action needed'}
    elif probability < 0.60:
        return {'label': 'Medium Risk', 'color': '🟡', 'bg_color': 'bg-yellow-500', 'action': 'Send engagement email'}
    elif probability < 0.80:
        return {'label': 'High Risk', 'color': '🟠', 'bg_color': 'bg-orange-500', 'action': 'Offer discount or assign success manager'}
    else:
        return {'label': 'Critical Risk', 'color': '🔴', 'bg_color': 'bg-red-600', 'action': 'Immediate intervention required!'}

def get_human_reasons(customer_data, shap_values, feature_names, top_n=3):
    """Convert SHAP values into human-readable reasons."""
    if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
        shap_flat = shap_values[0]
    else:
        shap_flat = shap_values
    
    impacts = list(zip(feature_names, shap_flat))
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    mapping = {
        'Contract_Month-to-month': 'Month-to-month contract (high churn risk)',
        'tenure': 'Very short tenure (new customer)',
        'MonthlyCharges': 'High monthly charges',
        'InternetService_Fiber optic': 'Fiber optic internet (higher churn)',
        'PaymentMethod_Electronic check': 'Using electronic checks (risky payment)',
        'PaperlessBilling_Yes': 'Paperless billing (detaches from service)',
        'OnlineSecurity_No': 'No online security service',
        'TechSupport_No': 'No tech support subscription',
        'StreamingTV_No': 'No streaming TV service',
        'Dependents_Yes': 'Has dependents (more likely to stay)',
        'Partner_Yes': 'Has a partner (more likely to stay)',
        'SeniorCitizen': 'Senior citizen status',
        'TotalCharges': 'Low total lifetime spend'
    }
    
    reasons = []
    for name, value in impacts[:top_n]:
        if value > 0.01:
            human_name = mapping.get(name, name.replace('_', ' '))
            reasons.append(human_name)
    return reasons

# ------------------------------
# 4. FastAPI App Setup
# ------------------------------
app = FastAPI(
    title="AI Customer Retention Platform API",
    description="Predict churn, get explanations, and recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 5. Pydantic Schemas
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
# 6. API Endpoints
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
    """
    try:
        # Convert input to dict
        input_dict = customer.dict()
        
        # Step 1: Preprocess (convert to one-hot encoded format)
        processed_df = preprocess_customer(input_dict)
        
        # Step 2: Scale the data
        scaled_data = scaler.transform(processed_df)
        
        # Step 3: Get prediction probability
        prob = model.predict_proba(scaled_data)[0][1]
        prob_percent = prob * 100
        
        # Step 4: Get risk category
        risk_info = get_risk_category(prob)
        
        # Step 5: Get SHAP reasons
        shap_vals = explainer.shap_values(scaled_data)
        reasons = get_human_reasons(processed_df, shap_vals, expected_features)
        
        # Step 6: Generate recommendation
        recommendation = risk_info['action']
        reasons_text = " ".join(reasons).lower()
        if "contract" in reasons_text:
            recommendation = "🎁 Offer a 1-year contract with a 15% discount"
        elif "charges" in reasons_text:
            recommendation = "💰 Offer a temporary 20% discount on next 3 bills"
        elif "tenure" in reasons_text:
            recommendation = "📧 Send welcome email series with premium tips"
        elif "support" in reasons_text or "security" in reasons_text:
            recommendation = "👤 Assign a dedicated Customer Success Manager"
        
        return PredictionResponse(
            probability=round(prob_percent, 2),
            risk_level=risk_info['label'],
            risk_color=risk_info['color'],
            reasons=reasons,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
def batch_predict(customers: List[CustomerInput]):
    """Predict for multiple customers."""
    results = []
    for customer in customers:
        input_dict = customer.dict()
        processed_df = preprocess_customer(input_dict)
        scaled_data = scaler.transform(processed_df)
        prob = model.predict_proba(scaled_data)[0][1]
        risk_info = get_risk_category(prob)
        results.append({
            "probability": round(prob * 100, 2),
            "risk_level": risk_info['label'],
            "risk_color": risk_info['color']
        })
    return {"results": results}
