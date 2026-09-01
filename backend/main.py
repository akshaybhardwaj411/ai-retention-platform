import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from supabase import create_client, Client

# ------------------------------
# 1. Load Models
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
    expected_features = json.load(f)

# ------------------------------
# 2. SHAP Load (with fallback)
# ------------------------------
shap_available = False
explainer = None

try:
    import shap
    try:
        explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
        shap_available = True
        print("✅ SHAP explainer loaded from disk.")
    except FileNotFoundError:
        explainer = shap.TreeExplainer(model)
        shap_available = True
        print("⚠️ SHAP explainer re-created from model (file not found).")
except Exception as e:
    shap_available = False
    explainer = None
    print(f"⚠️ SHAP not available (proceeding without explanations): {e}")

# ------------------------------
# 3. Supabase Connection
# ------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected successfully!")
    except Exception as e:
        supabase = None
        print(f"⚠️ Supabase connection failed: {e}")
else:
    supabase = None
    print("⚠️ Supabase environment variables not set.")

# ------------------------------
# 4. Preprocessing Function
# ------------------------------
def preprocess_customer(input_dict):
    default_data = {
        'gender': 'Male',
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
    
    for key, value in input_dict.items():
        default_data[key] = value
    
    df = pd.DataFrame([default_data])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[expected_features]
    return df_encoded

# ------------------------------
# 5. Helper Functions
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
# 6. FastAPI App with PROPER CORS
# ------------------------------
app = FastAPI(
    title="AI Customer Retention Platform API",
    description="Predict churn, get explanations, and recommendations",
    version="1.0.0"
)

# Allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 7. Schemas
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
# 8. Endpoints
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
        processed_df = preprocess_customer(input_dict)
        scaled_data = scaler.transform(processed_df)
        prob = model.predict_proba(scaled_data)[0][1]
        prob_percent = prob * 100
        
        risk_info = get_risk_category(prob)
        
        shap_vals = None
        if shap_available and explainer is not None:
            try:
                shap_vals = explainer.shap_values(scaled_data)
            except:
                shap_vals = None
        reasons = get_human_reasons(processed_df, shap_vals, expected_features)
        
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
        print(f"🔥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-and-save")
def predict_and_save(customer: CustomerInput, customer_name: str = None, customer_email: str = None):
    try:
        result = predict_churn(customer)
        customer_id = None
        if customer_email and supabase:
            existing = supabase.table("customers").select("*").eq("email", customer_email).execute()
            if existing.data:
                customer_id = existing.data[0]['id']
            else:
                new_customer = {
                    "name": customer_name or "Unknown",
                    "email": customer_email,
                    "gender": customer.gender,
                    "senior_citizen": customer.SeniorCitizen,
                    "partner": customer.Partner,
                    "dependents": customer.Dependents,
                    "tenure": customer.tenure,
                    "phone_service": customer.PhoneService,
                    "internet_service": customer.InternetService,
                    "contract": customer.Contract,
                    "payment_method": customer.PaymentMethod,
                    "monthly_charges": customer.MonthlyCharges,
                    "total_charges": customer.TotalCharges,
                    "last_login": "2026-09-01"
                }
                inserted = supabase.table("customers").insert(new_customer).execute()
                if inserted.data:
                    customer_id = inserted.data[0]['id']
        
        saved = False
        if supabase and customer_id:
            prediction_data = {
                "customer_id": customer_id,
                "churn_probability": result.probability,
                "risk_level": result.risk_level,
                "reasons": json.dumps(result.reasons),
                "recommendation": result.recommendation
            }
            supabase.table("predictions").insert(prediction_data).execute()
            saved = True
        
        return {
            **result.dict(),
            "customer_id": customer_id,
            "saved_to_db": saved
        }
    
    except Exception as e:
        print(f"🔥 ERROR in predict-and-save: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction and save failed: {str(e)}")
