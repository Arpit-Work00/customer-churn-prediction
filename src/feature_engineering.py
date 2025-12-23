"""
Feature Engineering Module for Customer Churn Prediction
=========================================================
This module handles encoding, scaling, and feature transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def get_categorical_columns(df):
    """
    Identify categorical columns in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        list: List of categorical column names
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_cols


def get_numerical_columns(df):
    """
    Identify numerical columns in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        list: List of numerical column names
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Remove target variable if present
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    return numerical_cols


def encode_categorical_features(df, method='label'):
    """
    Encode categorical features using Label Encoding or One-Hot Encoding.
    
    Args:
        df (pd.DataFrame): Input dataset
        method (str): 'label' for Label Encoding, 'onehot' for One-Hot Encoding
        
    Returns:
        pd.DataFrame: Dataset with encoded features
        dict: Dictionary of encoders for each column (for inverse transform)
    """
    df_encoded = df.copy()
    categorical_cols = get_categorical_columns(df_encoded)
    encoders = {}
    
    if method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        encoders = None  # One-hot doesn't need inverse transform typically
        
    return df_encoded, encoders


def scale_numerical_features(df, columns=None):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (list): Columns to scale. If None, scales all numerical columns.
        
    Returns:
        pd.DataFrame: Dataset with scaled features
        StandardScaler: Fitted scaler object
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = get_numerical_columns(df_scaled)
    
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled, scaler


def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using oversampling techniques.
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        method (str): 'smote' for SMOTE oversampling
        random_state (int): Random state for reproducibility
        
    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    print(f"\nOriginal class distribution:")
    print(f"  Class 0 (No Churn): {sum(y == 0)}")
    print(f"  Class 1 (Churn): {sum(y == 1)}")
    
    if method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
    print(f"\nResampled class distribution:")
    print(f"  Class 0 (No Churn): {sum(y_resampled == 0)}")
    print(f"  Class 1 (Churn): {sum(y_resampled == 1)}")
    
    return X_resampled, y_resampled


def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor


def prepare_features(df, target_col='Churn', use_smote=True):
    """
    Complete feature preparation pipeline.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        target_col (str): Name of target column
        use_smote (bool): Whether to apply SMOTE for class balancing
        
    Returns:
        X: Feature matrix (encoded)
        y: Target variable
        feature_names: List of feature names after encoding
    """
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Get column types
    categorical_cols = get_categorical_columns(X)
    numerical_cols = get_numerical_columns(X)
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical features using label encoding for tree models
    X_encoded, encoders = encode_categorical_features(X, method='label')
    
    # Note: Skipping scaling - tree-based models (Random Forest, XGBoost, Decision Tree)
    # don't require feature scaling and work fine with raw numerical values
    scaler = None
    
    # Handle class imbalance if requested
    if use_smote:
        X_balanced, y_balanced = handle_class_imbalance(X_encoded, y)
        return X_balanced, y_balanced, X_encoded.columns.tolist(), encoders, scaler
    
    return X_encoded, y, X_encoded.columns.tolist(), encoders, scaler


def get_feature_importance_names(encoders, numerical_cols):
    """
    Get meaningful feature names for interpretation.
    
    Args:
        encoders (dict): Dictionary of label encoders
        numerical_cols (list): List of numerical columns
        
    Returns:
        list: Feature names in order
    """
    feature_names = []
    
    # Add categorical column names
    if encoders:
        feature_names.extend(list(encoders.keys()))
    
    # Add numerical column names
    feature_names.extend(numerical_cols)
    
    return feature_names


if __name__ == "__main__":
    # Test the module
    from data_preprocessing import load_and_clean_data
    
    print("Loading data...")
    df = load_and_clean_data()
    
    print("\nPreparing features...")
    X, y, feature_names, encoders, scaler = prepare_features(df, use_smote=True)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {feature_names}")
