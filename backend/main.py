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
# 1. Load Models
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
    expected_features = json.load(f)

# Load SHAP explainer (if fails, we'll still run predictions without SHAP)
try:
    import shap
    explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    shap_available = True
except:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_available = True
    print("⚠️ SHAP explainer re-created from model.")
except Exception as e:
    shap_available = False
    explainer = None
    print(f"⚠️ SHAP not available: {e}")

# ------------------------------
# 2. BULLETPROOF PREPROCESSING
# ------------------------------
def preprocess_customer(input_dict):
    """
    Converts raw JSON into the exact feature set the model expects.
    Uses a default template to ensure ALL columns exist.
    """
    # 1. Define default values for ALL categorical columns
    # (All possible columns from Telco dataset, set to 'No' by default)
    default_data = {
        'gender': 'Male',               # Will be overwritten
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 0,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 0.0,
        'TotalCharges': 0.0
    }
    
    # 2. Override defaults with user input
    for key, value in input_dict.items():
        default_data[key] = value
    
    # 3. Create DataFrame from the merged data
    df = pd.DataFrame([default_data])
    
    # 4. Fix TotalCharges (convert to numeric, handle errors)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)  # Safe assignment (no inplace)
    
    # 5. Apply one-hot encoding to categorical columns
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 6. Ensure all expected columns are present (fill missing with 0)
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # 7. Reorder columns exactly as the model expects
    df_encoded = df_encoded[expected_features]
    
    return df_encoded

# ------------------------------
# 3. Helper Functions
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

def get_human_reasons(df_encoded, shap_values, feature_names, top_n=3):
    """Convert SHAP values into plain English reasons."""
    if shap_values is None:
        return ["AI explanation unavailable, but risk score is reliable."]
    
    try:
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
        return reasons if reasons else ["No specific churn drivers detected."]
    except:
        return ["AI explanation temporarily unavailable."]

# ------------------------------
# 4. FastAPI App
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
# 5. Schemas
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

class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    risk_color: str
    reasons: List[str]
    recommendation: str

# ------------------------------
# 6. Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"message": "AI Customer Retention Platform API is live!", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        input_dict = customer.dict()
        
        # Step 1: Preprocess using the bulletproof method
        processed_df = preprocess_customer(input_dict)
        
        # Step 2: Scale
        scaled_data = scaler.transform(processed_df)
        
        # Step 3: Predict probability
        prob = model.predict_proba(scaled_data)[0][1]
        prob_percent = prob * 100
        
        # Step 4: Risk
        risk_info = get_risk_category(prob)
        
        # Step 5: SHAP Reasons (with graceful fallback)
        shap_vals = None
        if shap_available and explainer is not None:
            try:
                shap_vals = explainer.shap_values(scaled_data)
            except:
                shap_vals = None
        reasons = get_human_reasons(processed_df, shap_vals, expected_features)
        
        # Step 6: Recommendation
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
        # Log the full error to Render logs
        print(f"🔥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
