from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk_category(probability: float) -> Dict[str, str]:
    """
    Convert churn probability (0.0 - 1.0) into a
    business-friendly risk category.
    """

    probability = float(
        np.clip(probability, 0.0, 1.0)
    )

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
            "action": "Send an engagement campaign",
        }

    if probability < 0.80:
        return {
            "label": "High Risk",
            "color": "🟠",
            "bg_color": "bg-orange-500",
            "action": (
                "Offer a personalized retention incentive "
                "or assign a customer success manager"
            ),
        }

    return {
        "label": "Critical Risk",
        "color": "🔴",
        "bg_color": "bg-red-600",
        "action": (
            "Immediate customer intervention required"
        ),
    }


# ============================================================
# FEATURE → HUMAN READABLE REASON
# ============================================================

FEATURE_REASON_MAP = {
    "Contract_Month-to-month":
        "Month-to-month contract may increase churn risk",

    "tenure":
        "Short customer tenure",

    "MonthlyCharges":
        "High monthly charges",

    "TotalCharges":
        "Low total lifetime spend",

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

    "MultipleLines_Yes":
        "Multiple phone lines",

    "OnlineBackup_No":
        "No online backup service",

    "DeviceProtection_No":
        "No device protection service",
}


# ============================================================
# SHAP NORMALIZATION
# ============================================================

def _normalize_shap_values(
    shap_values: Any,
    expected_feature_count: int,
) -> Optional[np.ndarray]:
    """
    Normalize SHAP output into a 1-D numpy array.

    Supports:
    - numpy arrays
    - list outputs
    - binary-class SHAP outputs
    - single-row outputs
    """

    if shap_values is None:
        return None

    try:
        values = shap_values

        # Some SHAP versions return a list.
        if isinstance(values, list):

            if len(values) == 0:
                return None

            # For binary classification, positive class
            # is usually the second element.
            if len(values) == 2:
                values = values[1]
            else:
                values = values[0]

        values = np.asarray(values)

        if values.size == 0:
            return None

        # Remove unnecessary dimensions.
        values = np.squeeze(values)

        # If still multidimensional, flatten carefully.
        if values.ndim > 1:

            if (
                values.ndim == 2
                and values.shape[0] == 1
            ):
                values = values[0]

            elif (
                values.ndim == 2
                and values.shape[1] == 1
            ):
                values = values[:, 0]

            else:
                values = values.reshape(-1)

        values = values.astype(float)

        # Ensure feature count matches.
        if len(values) > expected_feature_count:
            values = values[:expected_feature_count]

        elif len(values) < expected_feature_count:
            padded = np.zeros(
                expected_feature_count,
                dtype=float,
            )

            padded[:len(values)] = values
            values = padded

        return values

    except Exception:
        return None


# ============================================================
# HUMAN READABLE SHAP REASONS
# ============================================================

def get_human_reasons(
    customer_data: Optional[pd.DataFrame],
    shap_values: Any,
    feature_names: Sequence[str],
    top_n: int = 3,
) -> List[str]:
    """
    Convert SHAP feature impacts into human-readable
    churn-driving reasons.
    """

    if not feature_names:
        return [
            "Prediction explanation is unavailable."
        ]

    normalized_values = _normalize_shap_values(
        shap_values,
        len(feature_names),
    )

    if normalized_values is None:
        return [
            "Prediction explanation is currently unavailable."
        ]

    impacts = list(
        zip(
            feature_names,
            normalized_values,
        )
    )

    # Only positive SHAP values indicate features that
    # push the prediction towards churn.
    positive_impacts = [
        (name, float(value))
        for name, value in impacts
        if np.isfinite(value) and value > 0.01
    ]

    # Highest churn impact first.
    positive_impacts.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    reasons: List[str] = []

    for feature_name, _ in positive_impacts[:top_n]:

        human_reason = FEATURE_REASON_MAP.get(
            feature_name,
            feature_name.replace("_", " "),
        )

        if human_reason not in reasons:
            reasons.append(human_reason)

    if not reasons:
        return [
            "No strong individual churn drivers detected."
        ]

    return reasons


# ============================================================
# RETENTION RECOMMENDATION
# ============================================================

def generate_basic_recommendation(
    risk_info: Dict[str, str],
    reasons: List[str],
) -> str:
    """
    Generate a deterministic fallback recommendation.

    This is intentionally kept separate from the future
    Generative AI retention engine.
    """

    if not reasons:
        return risk_info["action"]

    reasons_text = " ".join(
        reasons
    ).lower()

    if "contract" in reasons_text:
        return (
            "Offer a longer-term contract with "
            "a personalized retention discount."
        )

    if "monthly charges" in reasons_text:
        return (
            "Offer a temporary discount or a "
            "lower-cost plan option."
        )

    if "tenure" in reasons_text:
        return (
            "Launch a personalized onboarding and "
            "engagement campaign."
        )

    if (
        "technical support" in reasons_text
        or "support" in reasons_text
        or "security" in reasons_text
    ):
        return (
            "Assign dedicated customer support and "
            "offer relevant service assistance."
        )

    if "payment" in reasons_text:
        return (
            "Offer an easier payment option and "
            "payment-related assistance."
        )

    return risk_info["action"]


# ============================================================
# COMPLETE PREDICTION + EXPLANATION
# ============================================================

def predict_with_explanation(
    customer_data_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    explainer: Any = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run churn prediction and generate an explainable result.

    Returns:
        probability
        risk_level
        risk_color
        reasons
        recommendation
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if customer_data_df is None:
        raise ValueError(
            "Customer data cannot be None."
        )

    if not isinstance(
        customer_data_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "customer_data_df must be a pandas DataFrame."
        )

    if customer_data_df.empty:
        raise ValueError(
            "Customer data cannot be empty."
        )

    if model is None:
        raise ValueError(
            "Machine learning model is not available."
        )

    if scaler is None:
        raise ValueError(
            "Scaler is not available."
        )

    if feature_names is None:
        feature_names = list(
            customer_data_df.columns
        )

    feature_names = list(
        feature_names
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        customer_data_df
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        scaled_data
    )

    if (
        probabilities is None
        or len(probabilities) == 0
    ):
        raise ValueError(
            "Model did not return prediction probabilities."
        )

    # Probability of churn = class 1.
    if probabilities.shape[1] >= 2:
        churn_probability = float(
            probabilities[0][1]
        )
    else:
        churn_probability = float(
            probabilities[0][0]
        )

    churn_probability = float(
        np.clip(
            churn_probability,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_info = get_risk_category(
        churn_probability
    )

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_values = None

    if explainer is not None:

        try:
            shap_values = (
                explainer.shap_values(
                    scaled_data
                )
            )

        except Exception:
            shap_values = None

    reasons = get_human_reasons(
        customer_data=customer_data_df,
        shap_values=shap_values,
        feature_names=feature_names,
        top_n=3,
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = (
        generate_basic_recommendation(
            risk_info=risk_info,
            reasons=reasons,
        )
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "probability": round(
            churn_probability * 100,
            2,
        ),
        "risk_level": risk_info["label"],
        "risk_color": risk_info["color"],
        "bg_color": risk_info["bg_color"],
        "reasons": reasons,
        "recommendation": recommendation,
    }
