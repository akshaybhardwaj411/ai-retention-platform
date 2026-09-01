import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from typing import List
from datetime import datetime

# ------------------------------
# DATABASE IMPORTS
# ------------------------------
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ------------------------------
# 1. Load Environment Variables
# ------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable not set!")

# ------------------------------
# 2. SQLAlchemy Setup
# ------------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ------------------------------
# 3. Database Models
# ------------------------------
class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    email = Column(String(100), unique=True, nullable=True)
    gender = Column(String(20))
    senior_citizen = Column(Integer)
    partner = Column(String(10))
    dependents = Column(String(10))
    tenure = Column(Integer)
    phone_service = Column(String(10))
    internet_service = Column(String(50))
    contract = Column(String(50))
    payment_method = Column(String(50))
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    churn_probability = Column(Float)
    risk_level = Column(String(50))
    reasons = Column(String(500))  # store as JSON string or comma-separated
    recommendation = Column(String(200))
    predicted_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------------
# 4. Load ML Models (same as before)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
    expected_features = json.load(f)

# SHAP loading (with fallback)
shap_available = False
explainer = None
try:
    import shap
    try:
        explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
        shap_available = True
    except:
        explainer = shap.TreeExplainer(model)
        shap_available = True
except:
    pass

# ------------------------------
# 5. Preprocessing & Helpers (exact same as before)
# ------------------------------
def preprocess_customer(input_dict):
    default_data = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 0, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
        'InternetService': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 0.0, 'TotalCharges': 0.0
    }
    for key, value in input_dict.items():
        default_data[key] = value
    df = pd.DataFrame([default_data])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[expected_features]

def get_risk_category(probability):
    if probability < 0.30:
        return {'label': 'Low Risk', 'color': '🟢', 'action': 'No immediate action'}
    elif probability < 0.60:
        return {'label': 'Medium Risk', 'color': '🟡', 'action': 'Send engagement email'}
    elif probability < 0.80:
        return {'label': 'High Risk', 'color': '🟠', 'action': 'Offer discount or assign manager'}
    else:
        return {'label': 'Critical Risk', 'color': '🔴', 'action': 'Immediate intervention!'}

def get_human_reasons(df_encoded, shap_values, feature_names, top_n=3):
    if shap_values is None:
        return ["Explanation unavailable."]
    try:
        if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
            shap_flat = shap_values[0]
        else:
            shap_flat = shap_values
        impacts = list(zip(feature_names, shap_flat))
        impacts.sort(key=lambda x: x[1], reverse=True)
        mapping = {
            'Contract_Month-to-month': 'Month-to-month contract',
            'tenure': 'Short tenure',
            'MonthlyCharges': 'High charges',
            'InternetService_Fiber optic': 'Fiber optic internet',
            'PaymentMethod_Electronic check': 'Electronic check payment'
        }
        reasons = []
        for name, value in impacts[:top_n]:
            if value > 0.01:
                human_name = mapping.get(name, name.replace('_', ' '))
                reasons.append(human_name)
        return reasons if reasons else ["No specific drivers detected."]
    except:
        return ["Explanation unavailable."]

# ------------------------------
# 6. FastAPI App
# ------------------------------
app = FastAPI(title="AI Retention Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 7. Pydantic Schemas
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

class CustomerCreate(BaseModel):
    name: str
    email: str
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
# 8. API Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"message": "AI Retention Platform with PostgreSQL", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput, db: Session = Depends(get_db)):
    try:
        input_dict = customer.dict()
        processed_df = preprocess_customer(input_dict)
        scaled_data = scaler.transform(processed_df)
        prob = model.predict_proba(scaled_data)[0][1]
        risk_info = get_risk_category(prob)
        
        shap_vals = None
        if shap_available and explainer:
            try:
                shap_vals = explainer.shap_values(scaled_data)
            except:
                pass
        reasons = get_human_reasons(processed_df, shap_vals, expected_features)
        
        recommendation = risk_info['action']
        reasons_text = " ".join(reasons).lower()
        if "contract" in reasons_text:
            recommendation = "🎁 Offer 1-year contract discount"
        elif "charges" in reasons_text:
            recommendation = "💰 Offer 20% discount"
        elif "tenure" in reasons_text:
            recommendation = "📧 Send welcome email series"
        
        # 📝 SAVE TO DATABASE
        # First, check if customer exists (by email/name - simplified)
        # For demo, we create a dummy customer record
        dummy_customer = Customer(
            gender=input_dict['gender'],
            senior_citizen=input_dict['SeniorCitizen'],
            partner=input_dict['Partner'],
            dependents=input_dict['Dependents'],
            tenure=input_dict['tenure'],
            phone_service=input_dict['PhoneService'],
            internet_service=input_dict['InternetService'],
            contract=input_dict['Contract'],
            payment_method=input_dict['PaymentMethod'],
            monthly_charges=input_dict['MonthlyCharges'],
            total_charges=input_dict['TotalCharges']
        )
        db.add(dummy_customer)
        db.commit()
        db.refresh(dummy_customer)
        
        # Save prediction
        pred_record = Prediction(
            customer_id=dummy_customer.id,
            churn_probability=round(prob * 100, 2),
            risk_level=risk_info['label'],
            reasons=", ".join(reasons),
            recommendation=recommendation
        )
        db.add(pred_record)
        db.commit()
        
        return PredictionResponse(
            probability=round(prob * 100, 2),
            risk_level=risk_info['label'],
            risk_color=risk_info['color'],
            reasons=reasons,
            recommendation=recommendation
        )
    except Exception as e:
        print(f"🔥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customers")
def list_customers(db: Session = Depends(get_db)):
    customers = db.query(Customer).all()
    return {"count": len(customers), "customers": customers}
