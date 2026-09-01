import json
import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from supabase import Client, create_client


# ============================================================
# 1. PATHS & MODEL LOADING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_required_model(filename: str) -> Any:
    """Load a required ML artifact and raise a clear startup error."""
    path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model file not found: {path}"
        )

    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model file '{filename}': {exc}"
        ) from exc


model = load_required_model("churn_model.pkl")
scaler = load_required_model("scaler.pkl")


FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

if not os.path.exists(FEATURE_NAMES_PATH):
    raise FileNotFoundError(
        f"Required feature file not found: {FEATURE_NAMES_PATH}"
    )

with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
    expected_features = json.load(file)


# ============================================================
# 2. SHAP EXPLAINABILITY
# ============================================================

shap_available = False
explainer = None

try:
    import shap

    shap_explainer_path = os.path.join(
        MODELS_DIR,
        "shap_explainer.pkl"
    )

    if os.path.exists(shap_explainer_path):
        explainer = joblib.load(shap_explainer_path)
        print("SHAP explainer loaded from disk.")
    else:
        explainer = shap.TreeExplainer(model)
        print("SHAP explainer recreated from model.")

    shap_available = True

except Exception as exc:
    shap_available = False
    explainer = None
    print(f"SHAP unavailable: {exc}")


# ============================================================
# 3. SUPABASE CONNECTION
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )
        print("Supabase connected successfully.")

    except Exception as exc:
        supabase = None
        print(f"Supabase connection failed: {exc}")

else:
    print("SUPABASE_URL or SUPABASE_KEY is not configured.")


# ============================================================
# 4. PREPROCESSING
# ============================================================

def preprocess_customer(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert customer input into the exact feature structure
    expected by the trained ML model.
    """

    default_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 0.0,
        "TotalCharges": 0.0,
    }

    default_data.update(input_dict)

    df = pd.DataFrame([default_data])

    # Numeric conversion
    df["SeniorCitizen"] = pd.to_numeric(
        df["SeniorCitizen"],
        errors="coerce"
    ).fillna(0)

    df["tenure"] = pd.to_numeric(
        df["tenure"],
        errors="coerce"
    ).fillna(0)

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    ).fillna(0)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    categorical_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    # Make sure all model features exist
    for feature in expected_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0

    # Remove unexpected columns and preserve model order
    df_encoded = df_encoded[expected_features]

    return df_encoded


# ============================================================
# 5. RISK CLASSIFICATION
# ============================================================

def get_risk_category(probability: float) -> Dict[str, str]:
    """
    Convert churn probability (0-1) into a business-friendly
    risk category.
    """

    if probability < 0.30:
        return {
            "label": "Low Risk",
            "color": "🟢",
            "bg_color": "bg-green-500",
            "action": "No immediate action needed",
        }

    if probability < 0.60:
        return {
            "label": "Medium Risk",
            "color": "🟡",
            "bg_color": "bg-yellow-500",
            "action": "Send engagement email",
        }

    if probability < 0.80:
        return {
            "label": "High Risk",
            "color": "🟠",
            "bg_color": "bg-orange-500",
            "action": "Offer discount or assign success manager",
        }

    return {
        "label": "Critical Risk",
        "color": "🔴",
        "bg_color": "bg-red-600",
        "action": "Immediate intervention required",
    }


# ============================================================
# 6. SHAP → HUMAN READABLE REASONS
# ============================================================

def get_human_reasons(
    shap_values: Any,
    feature_names: List[str],
    top_n: int = 3,
) -> List[str]:
    """
    Convert SHAP values into human-readable churn reasons.
    """

    if shap_values is None:
        return [
            "AI explanation is currently unavailable."
        ]

    try:
        shap_array = np.asarray(shap_values)

        # Handle shapes such as (1, n_features)
        if shap_array.ndim > 1:
            shap_flat = shap_array[0]
        else:
            shap_flat = shap_array

        impacts = list(
            zip(feature_names, shap_flat)
        )

        # Highest positive impact first
        impacts.sort(
            key=lambda item: float(item[1]),
            reverse=True
        )

        mapping = {
            "Contract_Month-to-month":
                "Month-to-month contract may increase churn risk",

            "tenure":
                "Short customer tenure",

            "MonthlyCharges":
                "High monthly charges",

            "InternetService_Fiber optic":
                "Fiber optic internet service",

            "PaymentMethod_Electronic check":
                "Electronic check payment method",

            "PaperlessBilling_Yes":
                "Paperless billing",

            "OnlineSecurity_No":
                "No online security service",

            "TechSupport_No":
                "No technical support subscription",

            "StreamingTV_No":
                "No streaming TV service",

            "StreamingMovies_No":
                "No streaming movies service",

            "Dependents_Yes":
                "Customer has dependents",

            "Partner_Yes":
                "Customer has a partner",

            "SeniorCitizen":
                "Senior citizen status",

            "TotalCharges":
                "Low total customer spend",
        }

        reasons = []

        for feature_name, impact in impacts:
            impact = float(impact)

            # Only positive churn-driving features
            if impact <= 0.01:
                continue

            human_reason = mapping.get(
                feature_name,
                feature_name.replace("_", " ")
            )

            reasons.append(human_reason)

            if len(reasons) >= top_n:
                break

        if not reasons:
            return [
                "No strong individual churn drivers detected."
            ]

        return reasons

    except Exception as exc:
        print(f"SHAP explanation error: {exc}")
        return [
            "AI explanation is temporarily unavailable."
        ]


# ============================================================
# 7. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Customer Retention Platform API",
    description=(
        "API for customer churn prediction, "
        "explainable AI and customer retention analytics."
    ),
    version="1.1.0",
)


# ============================================================
# 8. CORS
# ============================================================

frontend_url = os.environ.get(
    "FRONTEND_URL",
    "http://localhost:5173"
)

allowed_origins = [
    origin.strip()
    for origin in frontend_url.split(",")
    if origin.strip()
]

# Local development fallback
if not allowed_origins:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# 9. SCHEMAS
# ============================================================

class CustomerInput(BaseModel):
    model_config = ConfigDict(
        extra="ignore"
    )

    gender: str = "Male"
    SeniorCitizen: int = Field(
        default=0,
        ge=0,
        le=1
    )
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(
        default=0,
        ge=0
    )
    PhoneService: str = "No"
    InternetService: str = "No"
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(
        default=0.0,
        ge=0
    )
    TotalCharges: float = Field(
        default=0.0,
        ge=0
    )


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
    risk_color: str
    reasons: List[str]
    recommendation: str


# ============================================================
# 10. ANALYTICS
# ============================================================

@app.get("/analytics")
def get_analytics():
    """
    Fetch analytics from Supabase.

    NOTE:
    The churn trend calculation is intentionally kept compatible
    with the existing project until the exact historical schema
    is finalized.
    """

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not connected."
        )

    try:
        customers_response = (
            supabase
            .table("customers")
            .select("*")
            .execute()
        )

        customers = customers_response.data or []

        predictions_response = (
            supabase
            .table("predictions")
            .select("*")
            .execute()
        )

        predictions = predictions_response.data or []

        total_customers = len(customers)

        latest_predictions: Dict[Any, Dict[str, Any]] = {}

        for prediction in predictions:
            customer_id = prediction.get("customer_id")
            predicted_at = prediction.get("predicted_at")

            if customer_id is None:
                continue

            if customer_id not in latest_predictions:
                latest_predictions[customer_id] = prediction
                continue

            previous_date = latest_predictions[
                customer_id
            ].get("predicted_at")

            if (
                predicted_at
                and previous_date
                and predicted_at > previous_date
            ):
                latest_predictions[
                    customer_id
                ] = prediction

        high_risk_count = 0
        critical_risk_count = 0
        total_revenue = 0.0
        active_count = 0

        customers_by_id = {
            customer.get("id"): customer
            for customer in customers
            if customer.get("id") is not None
        }

        for customer in customers:
            total_revenue += float(
                customer.get("total_charges") or 0
            )

            if (customer.get("tenure") or 0) > 0:
                active_count += 1

            customer_id = customer.get("id")

            prediction = latest_predictions.get(
                customer_id
            )

            if prediction:
                probability = float(
                    prediction.get(
                        "churn_probability",
                        0
                    )
                )

                # Database may store 0-1 or 0-100
                if probability <= 1:
                    probability *= 100

                if probability >= 80:
                    critical_risk_count += 1
                    high_risk_count += 1

                elif probability >= 60:
                    high_risk_count += 1

        # Existing project-compatible trend.
        # We will replace this with actual historical data
        # once the prediction/event schema is finalized.
        churn_trend = []

        end_date = datetime.now()

        for i in range(5, -1, -1):
            month_date = (
                end_date -
                timedelta(days=30 * i)
            )

            month_key = month_date.strftime("%b")

            churn_trend.append(
                {
                    "month": month_key,
                    "churn": max(
                        0,
                        30 + i * 2 +
                        np.random.randint(-5, 5)
                    ),
                }
            )

        revenue_loss = 0.0

        for customer_id, prediction in latest_predictions.items():

            probability = float(
                prediction.get(
                    "churn_probability",
                    0
                )
            )

            if probability <= 1:
                probability *= 100

            if probability >= 60:

                customer = customers_by_id.get(
                    customer_id
                )

                if customer:
                    monthly_charges = float(
                        customer.get(
                            "monthly_charges",
                            0
                        ) or 0
                    )

                    revenue_loss += (
                        monthly_charges * 3
                    )

        retention_rate = (
            active_count / total_customers * 100
            if total_customers
            else 0
        )

        return {
            "totalCustomers": total_customers,
            "activeCustomers": active_count,
            "highRisk": high_risk_count,
            "criticalRisk": critical_risk_count,
            "retentionRate": round(
                retention_rate,
                2
            ),
            "revenue": round(
                total_revenue,
                2
            ),
            "revenueLoss": round(
                revenue_loss,
                2
            ),
            "churnTrend": churn_trend,
            "predictionsCount": len(predictions),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Analytics error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Unable to load analytics: {exc}"
        )


# ============================================================
# 11. CUSTOMER ENDPOINTS
# ============================================================

@app.get("/customers")
def get_all_customers():
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not connected."
        )

    try:
        response = (
            supabase
            .table("customers")
            .select("*")
            .execute()
        )

        return {
            "customers": response.data or []
        }

    except Exception as exc:
        print("Get customers error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Unable to fetch customers: {exc}"
        )


@app.get("/customers/{customer_id}")
def get_customer_by_id(customer_id: int):

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not connected."
        )

    try:
        response = (
            supabase
            .table("customers")
            .select("*")
            .eq("id", customer_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Customer not found."
            )

        return response.data[0]

    except HTTPException:
        raise

    except Exception as exc:
        print("Get customer error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Unable to fetch customer: {exc}"
        )


@app.get("/customers/{customer_id}/predictions")
def get_customer_predictions(customer_id: int):

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not connected."
        )

    try:
        response = (
            supabase
            .table("predictions")
            .select("*")
            .eq("customer_id", customer_id)
            .order(
                "predicted_at",
                desc=True
            )
            .execute()
        )

        return {
            "predictions": response.data or []
        }

    except Exception as exc:
        print("Get customer predictions error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to fetch customer predictions: "
                f"{exc}"
            )
        )


# ============================================================
# 12. ROOT & HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "message": (
            "AI Customer Retention Platform API is live!"
        ),
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "shap_available": shap_available,
        "database_connected": supabase is not None,
    }


# ============================================================
# 13. CHURN PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(
    customer: CustomerInput
):

    try:
        input_dict = customer.model_dump()

        processed_df = preprocess_customer(
            input_dict
        )

        scaled_data = scaler.transform(
            processed_df
        )

        prediction_probability = (
            model.predict_proba(
                scaled_data
            )[0][1]
        )

        probability = float(
            prediction_probability
        )

        probability_percent = probability * 100

        risk_info = get_risk_category(
            probability
        )

        shap_values = None

        if (
            shap_available
            and explainer is not None
        ):
            try:
                shap_values = (
                    explainer.shap_values(
                        scaled_data
                    )
                )

            except Exception as exc:
                print(
                    f"SHAP calculation failed: {exc}"
                )
                shap_values = None

        reasons = get_human_reasons(
            shap_values,
            expected_features
        )

        recommendation = risk_info["action"]

        reasons_text = (
            " ".join(reasons)
            .lower()
        )

        # Existing business recommendation logic.
        # Later this will be moved to the AI Retention Engine.
        if "contract" in reasons_text:
            recommendation = (
                "Offer a 1-year contract "
                "with a 15% discount."
            )

        elif "charges" in reasons_text:
            recommendation = (
                "Offer a temporary 20% discount "
                "on the next 3 bills."
            )

        elif "tenure" in reasons_text:
            recommendation = (
                "Send a personalized onboarding "
                "and engagement campaign."
            )

        elif (
            "support" in reasons_text
            or "security" in reasons_text
        ):
            recommendation = (
                "Assign a dedicated "
                "Customer Success Manager."
            )

        return PredictionResponse(
            probability=round(
                probability_percent,
                2
            ),
            risk_level=risk_info["label"],
            risk_color=risk_info["color"],
            reasons=reasons,
            recommendation=recommendation,
        )

    except HTTPException:
        raise

    except Exception as exc:
        print("Prediction error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )


# ============================================================
# 14. PREDICT AND SAVE
# ============================================================

@app.post("/predict-and-save")
def predict_and_save(
    customer: CustomerInput,
    customer_name: Optional[str] = None,
    customer_email: Optional[str] = None,
):

    try:
        result = predict_churn(customer)

        customer_id = None

        # ----------------------------------------------------
        # Save / find customer
        # ----------------------------------------------------

        if customer_email and supabase:

            existing = (
                supabase
                .table("customers")
                .select("*")
                .eq(
                    "email",
                    customer_email
                )
                .execute()
            )

            if existing.data:

                customer_id = (
                    existing.data[0]["id"]
                )

            else:

                new_customer = {
                    "name": (
                        customer_name
                        or "Unknown"
                    ),
                    "email": customer_email,
                    "gender": customer.gender,
                    "senior_citizen": (
                        customer.SeniorCitizen
                    ),
                    "partner": customer.Partner,
                    "dependents": (
                        customer.Dependents
                    ),
                    "tenure": customer.tenure,
                    "phone_service": (
                        customer.PhoneService
                    ),
                    "internet_service": (
                        customer.InternetService
                    ),
                    "contract": customer.Contract,
                    "payment_method": (
                        customer.PaymentMethod
                    ),
                    "monthly_charges": (
                        customer.MonthlyCharges
                    ),
                    "total_charges": (
                        customer.TotalCharges
                    ),
                    "last_login": (
                        datetime.now()
                        .strftime("%Y-%m-%d")
                    ),
                }

                inserted = (
                    supabase
                    .table("customers")
                    .insert(new_customer)
                    .execute()
                )

                if inserted.data:
                    customer_id = (
                        inserted.data[0]["id"]
                    )

        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        saved = False

        if supabase and customer_id:

            prediction_data = {
                "customer_id": customer_id,
                "churn_probability": (
                    result.probability
                ),
                "risk_level": (
                    result.risk_level
                ),
                "reasons": json.dumps(
                    result.reasons
                ),
                "recommendation": (
                    result.recommendation
                ),
            }

            (
                supabase
                .table("predictions")
                .insert(prediction_data)
                .execute()
            )

            saved = True

        return {
            **result.model_dump(),
            "customer_id": customer_id,
            "saved_to_db": saved,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Predict-and-save error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction and save failed: "
                f"{exc}"
            )
)
