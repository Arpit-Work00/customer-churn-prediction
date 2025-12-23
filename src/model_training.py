"""
Model Training Module for Customer Churn Prediction
=====================================================
This module trains multiple ML models and evaluates their performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
VIZ_PATH = os.path.join(PROJECT_ROOT, 'visualizations')


def ensure_directories():
    """Create necessary directories."""
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(VIZ_PATH, exist_ok=True)


def get_models():
    """
    Get dictionary of models to train.
    
    Returns:
        dict: Model name -> model instance
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=2  # Handle imbalance
        )
    }
    return models


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train a single model and return metrics.
    
    Args:
        model: sklearn-compatible model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        dict: Model metrics
    """
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def train_all_models(X, y, test_size=0.2, random_state=42):
    """
    Train all models and compare performance.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Test split ratio
        random_state: Random state for reproducibility
        
    Returns:
        dict: Model name -> (model, metrics)
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("🤖 MODEL TRAINING - CUSTOMER CHURN PREDICTION")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Churn rate (train): {y_train.mean()*100:.1f}%")
    print(f"   Churn rate (test): {y_test.mean()*100:.1f}%")
    
    models = get_models()
    results = {}
    
    print("\n🔄 Training Models...\n")
    
    for name, model in models.items():
        print(f"   Training {name}...", end=" ")
        metrics = train_model(model, X_train, y_train, X_test, y_test)
        results[name] = (model, metrics)
        print(f"✓ (Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f})")
    
    return results, (X_train, X_test, y_train, y_test)


def display_results_table(results):
    """
    Display model comparison table.
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "=" * 60)
    print("📈 MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Create comparison DataFrame
    comparison = []
    for name, (model, metrics) in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison)
    print("\n", df_comparison.to_string(index=False))
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x][1]['roc_auc'])
    best_metrics = results[best_model_name][1]
    
    print("\n" + "-" * 60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
    print("-" * 60)
    
    return df_comparison, best_model_name


def plot_confusion_matrices(results, y_test, save=True):
    """
    Plot confusion matrices for all models.
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        save: Whether to save the plot
    """
    ensure_directories()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, (model, metrics)) in enumerate(results.items()):
        cm = confusion_matrix(y_test, metrics['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        axes[idx].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.3f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '08_confusion_matrices.png'), dpi=150, bbox_inches='tight')
        print(f"\nSaved: 08_confusion_matrices.png")
    
    plt.close()


def plot_roc_curves(results, y_test, save=True):
    """
    Plot ROC curves for all models.
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        save: Whether to save the plot
    """
    ensure_directories()
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (name, (model, metrics)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '09_roc_curves.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 09_roc_curves.png")
    
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save=True):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        save: Whether to save the plot
    """
    ensure_directories()
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        print(f"Feature importance not available for {model_name}")
        return
    
    # Create DataFrame and sort
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    # Plot top 15 features
    top_features = feat_imp.tail(15)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
    plt.barh(top_features['Feature'], top_features['Importance'], color=colors, edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 15 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '10_feature_importance.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 10_feature_importance.png")
    
    plt.close()
    
    return feat_imp


def save_best_model(results, feature_names, encoders=None, scaler=None):
    """
    Save the best performing model and associated objects.
    
    Args:
        results: Dictionary of model results
        feature_names: List of feature names
        encoders: Dictionary of label encoders
        scaler: Fitted scaler
        
    Returns:
        str: Path to saved model
    """
    ensure_directories()
    
    # Find best model based on ROC-AUC
    best_model_name = max(results.keys(), key=lambda x: results[x][1]['roc_auc'])
    best_model, best_metrics = results[best_model_name]
    
    # Create model package
    model_package = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_names': feature_names,
        'encoders': encoders,
        'scaler': scaler,
        'metrics': {
            'accuracy': best_metrics['accuracy'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1_score': best_metrics['f1_score'],
            'roc_auc': best_metrics['roc_auc']
        }
    }
    
    # Save model
    model_path = os.path.join(MODELS_PATH, 'best_model.pkl')
    joblib.dump(model_package, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"   Model: {best_model_name}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    return model_path


def load_model(model_path=None):
    """
    Load a saved model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        dict: Model package
    """
    if model_path is None:
        model_path = os.path.join(MODELS_PATH, 'best_model.pkl')
    
    model_package = joblib.load(model_path)
    return model_package


def run_training_pipeline():
    """
    Run the complete training pipeline.
    
    Returns:
        dict: Training results
    """
    # Import local modules
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_and_clean_data
    from feature_engineering import prepare_features
    
    # Load and prepare data
    print("📁 Loading data...")
    df = load_and_clean_data()
    
    print("\n🔧 Preparing features...")
    X, y, feature_names, encoders, scaler = prepare_features(df, use_smote=False)
    
    # Train models
    results, (X_train, X_test, y_train, y_test) = train_all_models(X, y)
    
    # Display results
    comparison_df, best_model_name = display_results_table(results)
    
    # Generate plots
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    
    # Feature importance for best model
    best_model = results[best_model_name][0]
    plot_feature_importance(best_model, feature_names, best_model_name)
    
    # Save best model
    model_path = save_best_model(results, feature_names, encoders, scaler)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    
    return {
        'results': results,
        'best_model_name': best_model_name,
        'comparison_df': comparison_df,
        'model_path': model_path
    }


if __name__ == "__main__":
    training_results = run_training_pipeline()
