"""
Data Preprocessing Module for Customer Churn Prediction
========================================================
This module handles loading, cleaning, and preparing the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'telco_customer_churn.csv')


def create_sample_dataset():
    """
    Create a sample Telco Customer Churn dataset for demonstration.
    This mimics the structure of the real Kaggle dataset.
    
    Returns:
        pd.DataFrame: Sample churn dataset
    """
    np.random.seed(42)
    n_samples = 7043  # Same as original dataset
    
    # Generate customer IDs
    customer_ids = [f'{i:04d}-{np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5)}' 
                   for i in range(1, n_samples + 1)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Tenure (months with company)
    tenure = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    # Multiple lines depends on phone service
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.42, 0.58]))
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
    
    # Internet-dependent services
    def get_internet_dependent(internet_svc):
        if internet_svc == 'No':
            return 'No internet service'
        return np.random.choice(['Yes', 'No'], p=[0.44, 0.56])
    
    online_security = [get_internet_dependent(i) for i in internet_service]
    online_backup = [get_internet_dependent(i) for i in internet_service]
    device_protection = [get_internet_dependent(i) for i in internet_service]
    tech_support = [get_internet_dependent(i) for i in internet_service]
    streaming_tv = [get_internet_dependent(i) for i in internet_service]
    streaming_movies = [get_internet_dependent(i) for i in internet_service]
    
    # Contract and billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples, p=[0.34, 0.23, 0.22, 0.21])
    
    # Charges
    monthly_charges = np.random.uniform(18, 120, n_samples).round(2)
    total_charges = (monthly_charges * tenure).round(2)
    
    # Some total charges are blank (converted to string to mimic real data issue)
    total_charges_str = total_charges.astype(str)
    blank_indices = np.random.choice(n_samples, 11, replace=False)
    total_charges_str[blank_indices] = ' '
    
    # Churn - influenced by various factors
    churn_prob = np.zeros(n_samples)
    
    # Higher churn for month-to-month contracts
    churn_prob += np.where(np.array(contract) == 'Month-to-month', 0.3, 0)
    
    # Higher churn for fiber optic (often due to pricing)
    churn_prob += np.where(np.array(internet_service) == 'Fiber optic', 0.15, 0)
    
    # Lower churn for longer tenure
    churn_prob -= tenure * 0.005
    
    # Higher churn for electronic check payment
    churn_prob += np.where(np.array(payment_method) == 'Electronic check', 0.1, 0)
    
    # Higher churn for higher monthly charges
    churn_prob += (monthly_charges - 50) * 0.002
    
    # Lower churn with security/support services
    churn_prob -= np.where(np.array(online_security) == 'Yes', 0.1, 0)
    churn_prob -= np.where(np.array(tech_support) == 'Yes', 0.1, 0)
    
    # Normalize and add randomness
    churn_prob = np.clip(churn_prob + np.random.uniform(-0.1, 0.1, n_samples), 0.05, 0.75)
    churn = np.where(np.random.random(n_samples) < churn_prob, 'Yes', 'No')
    
    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
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
        'TotalCharges': total_charges_str,
        'Churn': churn
    })
    
    return df


def load_data(file_path=None):
    """
    Load the customer churn dataset.
    
    Args:
        file_path (str): Path to the CSV file. If None, uses default path.
        
    Returns:
        pd.DataFrame: Raw dataset
    """
    if file_path is None:
        file_path = DATA_PATH
    
    # Check if file exists, if not create sample data
    if not os.path.exists(file_path):
        print("Dataset not found. Creating sample dataset...")
        df = create_sample_dataset()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Sample dataset saved to {file_path}")
    else:
        df = pd.read_csv(file_path)
    
    return df


def clean_data(df):
    """
    Clean the customer churn dataset.
    
    Steps:
    1. Handle missing values in TotalCharges
    2. Convert data types
    3. Drop customerID (not useful for prediction)
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_clean = df.copy()
    
    # TotalCharges has some blank values - convert to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with MonthlyCharges * tenure
    # (logical imputation based on domain knowledge)
    missing_mask = df_clean['TotalCharges'].isna()
    df_clean.loc[missing_mask, 'TotalCharges'] = (
        df_clean.loc[missing_mask, 'MonthlyCharges'] * 
        df_clean.loc[missing_mask, 'tenure']
    )
    
    # For customers with 0 tenure, set TotalCharges to MonthlyCharges
    zero_tenure_mask = df_clean['tenure'] == 0
    df_clean.loc[zero_tenure_mask & df_clean['TotalCharges'].isna(), 'TotalCharges'] = \
        df_clean.loc[zero_tenure_mask & df_clean['TotalCharges'].isna(), 'MonthlyCharges']
    
    # Drop customerID - not useful for prediction
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
    
    # Convert Churn to binary (for modeling)
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    return df_clean


def load_and_clean_data(file_path=None):
    """
    Load and clean data in one step.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis
    """
    df = load_data(file_path)
    df_clean = clean_data(df)
    return df_clean


def get_feature_info():
    """
    Get information about all features in the dataset.
    
    Returns:
        dict: Feature descriptions
    """
    features = {
        'customerID': 'Unique identifier for each customer',
        'gender': 'Customer gender (Male/Female)',
        'SeniorCitizen': 'Whether customer is a senior citizen (0/1)',
        'Partner': 'Whether customer has a partner (Yes/No)',
        'Dependents': 'Whether customer has dependents (Yes/No)',
        'tenure': 'Number of months customer has stayed with company',
        'PhoneService': 'Whether customer has phone service (Yes/No)',
        'MultipleLines': 'Whether customer has multiple lines (Yes/No/No phone service)',
        'InternetService': 'Type of internet service (DSL/Fiber optic/No)',
        'OnlineSecurity': 'Whether customer has online security add-on',
        'OnlineBackup': 'Whether customer has online backup add-on',
        'DeviceProtection': 'Whether customer has device protection add-on',
        'TechSupport': 'Whether customer has tech support add-on',
        'StreamingTV': 'Whether customer has streaming TV add-on',
        'StreamingMovies': 'Whether customer has streaming movies add-on',
        'Contract': 'Type of contract (Month-to-month/One year/Two year)',
        'PaperlessBilling': 'Whether customer has paperless billing (Yes/No)',
        'PaymentMethod': 'Payment method used by customer',
        'MonthlyCharges': 'Monthly charge amount in dollars',
        'TotalCharges': 'Total charges to date in dollars',
        'Churn': 'Whether customer churned (Yes/No) - TARGET VARIABLE'
    }
    return features


if __name__ == "__main__":
    # Test the module
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
