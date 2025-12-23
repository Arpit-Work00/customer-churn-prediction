"""
Utility Functions for Customer Churn Prediction
================================================
This module contains helper functions used across the project.
"""

import pandas as pd
import numpy as np
import os


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_customer_profile(
    gender='Male',
    senior_citizen=0,
    partner='Yes',
    dependents='No',
    tenure=12,
    phone_service='Yes',
    multiple_lines='No',
    internet_service='Fiber optic',
    online_security='No',
    online_backup='No',
    device_protection='No',
    tech_support='No',
    streaming_tv='No',
    streaming_movies='No',
    contract='Month-to-month',
    paperless_billing='Yes',
    payment_method='Electronic check',
    monthly_charges=70.0,
    total_charges=840.0
):
    """
    Create a customer profile dictionary for prediction.
    
    Args:
        Various customer attributes
        
    Returns:
        dict: Customer profile ready for prediction
    """
    return {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }


def format_prediction_result(prediction, probability):
    """
    Format prediction result for display.
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Churn probability
        
    Returns:
        dict: Formatted result
    """
    if prediction == 1:
        status = "⚠️ HIGH RISK - Customer Likely to Churn"
        risk_level = "High"
        color = "red"
    else:
        if probability > 0.3:
            status = "⚡ MEDIUM RISK - Monitor Customer"
            risk_level = "Medium"
            color = "orange"
        else:
            status = "✅ LOW RISK - Customer Likely to Stay"
            risk_level = "Low"
            color = "green"
    
    return {
        'prediction': prediction,
        'probability': probability,
        'status': status,
        'risk_level': risk_level,
        'color': color,
        'churn_text': 'Yes' if prediction == 1 else 'No'
    }


def get_model_explanations():
    """
    Get explanations for why each model was chosen.
    
    Returns:
        dict: Model name -> explanation
    """
    return {
        'Logistic Regression': """
        **Why Logistic Regression?**
        - Simple, interpretable baseline model
        - Works well for binary classification
        - Provides probability estimates
        - Fast to train and predict
        - Good for understanding feature impacts through coefficients
        """,
        
        'Decision Tree': """
        **Why Decision Tree?**
        - Highly interpretable (can visualize the tree)
        - No feature scaling required
        - Handles non-linear relationships
        - Easy to explain to stakeholders
        - Can capture complex decision boundaries
        """,
        
        'Random Forest': """
        **Why Random Forest?**
        - Ensemble of decision trees (reduces overfitting)
        - Robust to outliers and noise
        - Provides feature importance rankings
        - Handles high-dimensional data well
        - Generally very accurate
        """,
        
        'XGBoost': """
        **Why XGBoost?**
        - State-of-the-art gradient boosting algorithm
        - Handles imbalanced data well
        - Built-in regularization
        - Often wins ML competitions
        - Very high predictive accuracy
        """
    }


def get_retention_recommendations(customer_profile, probability):
    """
    Generate retention recommendations based on customer profile.
    
    Args:
        customer_profile: Dictionary of customer attributes
        probability: Churn probability
        
    Returns:
        list: Recommendations to reduce churn risk
    """
    recommendations = []
    
    if customer_profile.get('Contract') == 'Month-to-month':
        recommendations.append("📋 Offer long-term contract with discount (1-2 years)")
    
    if customer_profile.get('PaymentMethod') == 'Electronic check':
        recommendations.append("💳 Encourage auto-pay enrollment (credit/bank transfer)")
    
    if customer_profile.get('OnlineSecurity') == 'No' and customer_profile.get('InternetService') != 'No':
        recommendations.append("🔐 Offer free trial of Online Security service")
    
    if customer_profile.get('TechSupport') == 'No' and customer_profile.get('InternetService') != 'No':
        recommendations.append("🛠️ Offer complimentary Tech Support for 3 months")
    
    if customer_profile.get('tenure', 0) < 12:
        recommendations.append("🎁 Provide loyalty rewards for new customers")
    
    if customer_profile.get('MonthlyCharges', 0) > 80:
        recommendations.append("💰 Review pricing - consider personalized discount")
    
    if probability > 0.5:
        recommendations.append("📞 Schedule proactive customer success call")
        recommendations.append("⭐ Offer premium support as retention gesture")
    
    if not recommendations:
        recommendations.append("✅ Customer appears satisfied - continue current service")
    
    return recommendations
