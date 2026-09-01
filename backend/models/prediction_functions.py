
import joblib
import numpy as np
import pandas as pd
import shap

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

def predict_with_explanation(customer_data_df, model, scaler, explainer, feature_names):
    scaled_data = scaler.transform(customer_data_df)
    prob = model.predict_proba(scaled_data)[0][1]
    risk_info = get_risk_category(prob)
    shap_vals = explainer.shap_values(scaled_data)
    reasons = get_human_reasons(customer_data_df, shap_vals, feature_names)

    recommendation = risk_info['action']
    reasons_text = " ".join(reasons).lower()
    if "contract" in reasons_text:
        recommendation = "🎁 Offer a 1-year contract with a 15% discount"
    elif "charges" in reasons_text:
        recommendation = "💰 Offer a temporary 20% discount on next 3 bills"
    elif "tenure" in reasons_text:
        recommendation = "📧 Send welcome email series with premium tips"
    elif "support" in reasons_text:
        recommendation = "👤 Assign a dedicated Customer Success Manager"

    return {
        'probability': round(prob * 100, 2),
        'risk_level': risk_info['label'],
        'risk_color': risk_info['color'],
        'reasons': reasons,
        'recommendation': recommendation
    }
