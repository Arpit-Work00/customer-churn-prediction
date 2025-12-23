"""
Exploratory Data Analysis Module for Customer Churn Prediction
===============================================================
This module creates visualizations and insights from the churn dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_PATH = os.path.join(PROJECT_ROOT, 'visualizations')


def ensure_viz_directory():
    """Create visualizations directory if it doesn't exist."""
    os.makedirs(VIZ_PATH, exist_ok=True)


def plot_churn_distribution(df, save=True):
    """
    Visualize the distribution of churned vs non-churned customers.
    
    Args:
        df (pd.DataFrame): Dataset with Churn column
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['No Churn', 'Churn']
    
    axes[0].bar(labels, churn_counts.values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    
    # Add count labels on bars
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontsize=12, fontweight='bold')
    
    # Pie chart
    axes[1].pie(churn_counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0, 0.05), shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Churn Rate Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '01_churn_distribution.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 01_churn_distribution.png")
    
    plt.close()
    
    # Print insights
    churn_rate = (churn_counts[1] / churn_counts.sum()) * 100
    print(f"\n📊 INSIGHT: Churn Distribution")
    print(f"   - Total Customers: {len(df)}")
    print(f"   - Churned: {churn_counts[1]} ({churn_rate:.1f}%)")
    print(f"   - Not Churned: {churn_counts[0]} ({100-churn_rate:.1f}%)")
    print(f"   - The dataset shows class imbalance - need to handle this during modeling!")


def plot_churn_by_tenure(df, save=True):
    """
    Analyze churn patterns based on customer tenure.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparing tenure for churned vs non-churned
    df[df['Churn'] == 0]['tenure'].hist(ax=axes[0], bins=30, alpha=0.7, 
                                         label='No Churn', color='#2ecc71', edgecolor='black')
    df[df['Churn'] == 1]['tenure'].hist(ax=axes[0], bins=30, alpha=0.7, 
                                         label='Churn', color='#e74c3c', edgecolor='black')
    axes[0].set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tenure (months)', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].legend()
    
    # Box plot
    df_plot = df.copy()
    df_plot['Churn_Label'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    sns.boxplot(x='Churn_Label', y='tenure', data=df_plot, ax=axes[1], 
                palette=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Tenure Comparison: Churned vs Non-Churned', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Churn Status', fontsize=12)
    axes[1].set_ylabel('Tenure (months)', fontsize=12)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '02_churn_by_tenure.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 02_churn_by_tenure.png")
    
    plt.close()
    
    # Print insights
    avg_tenure_churn = df[df['Churn'] == 1]['tenure'].mean()
    avg_tenure_no_churn = df[df['Churn'] == 0]['tenure'].mean()
    print(f"\n📊 INSIGHT: Tenure Analysis")
    print(f"   - Avg tenure (Churned): {avg_tenure_churn:.1f} months")
    print(f"   - Avg tenure (Not Churned): {avg_tenure_no_churn:.1f} months")
    print(f"   - Customers with shorter tenure are MORE likely to churn!")


def plot_churn_by_contract(df, save=True):
    """
    Analyze churn patterns based on contract type.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Churn rate by contract type
    contract_churn = df.groupby('Contract')['Churn'].mean() * 100
    contract_order = ['Month-to-month', 'One year', 'Two year']
    contract_churn = contract_churn.reindex(contract_order)
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = axes[0].bar(contract_churn.index, contract_churn.values, color=colors, 
                       edgecolor='black', linewidth=1.2)
    axes[0].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Contract Type', fontsize=12)
    axes[0].set_ylabel('Churn Rate (%)', fontsize=12)
    axes[0].set_ylim(0, 50)
    
    # Add percentage labels
    for bar, val in zip(bars, contract_churn.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Stacked bar chart showing counts
    contract_counts = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
    contract_counts = contract_counts.reindex(contract_order)
    contract_counts.columns = ['No Churn', 'Churn']
    contract_counts.plot(kind='bar', stacked=True, ax=axes[1], 
                        color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.2)
    axes[1].set_title('Customer Distribution by Contract', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Contract Type', fontsize=12)
    axes[1].set_ylabel('Number of Customers', fontsize=12)
    axes[1].legend(title='Status')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '03_churn_by_contract.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 03_churn_by_contract.png")
    
    plt.close()
    
    print(f"\n📊 INSIGHT: Contract Type Analysis")
    print(f"   - Month-to-month contracts have the HIGHEST churn rate!")
    print(f"   - Two-year contracts have the LOWEST churn rate")
    print(f"   - Recommendation: Incentivize longer contracts to reduce churn")


def plot_churn_by_charges(df, save=True):
    """
    Analyze churn patterns based on monthly and total charges.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Monthly charges distribution
    df_plot = df.copy()
    df_plot['Churn_Label'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    
    sns.kdeplot(data=df[df['Churn'] == 0], x='MonthlyCharges', ax=axes[0], 
                label='No Churn', color='#2ecc71', fill=True, alpha=0.5)
    sns.kdeplot(data=df[df['Churn'] == 1], x='MonthlyCharges', ax=axes[0], 
                label='Churn', color='#e74c3c', fill=True, alpha=0.5)
    axes[0].set_title('Monthly Charges Distribution by Churn', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Monthly Charges ($)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()
    
    # Box plot for monthly charges
    sns.boxplot(x='Churn_Label', y='MonthlyCharges', data=df_plot, ax=axes[1],
                palette=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Monthly Charges: Churned vs Non-Churned', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Churn Status', fontsize=12)
    axes[1].set_ylabel('Monthly Charges ($)', fontsize=12)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '04_churn_by_charges.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 04_churn_by_charges.png")
    
    plt.close()
    
    avg_charges_churn = df[df['Churn'] == 1]['MonthlyCharges'].mean()
    avg_charges_no_churn = df[df['Churn'] == 0]['MonthlyCharges'].mean()
    print(f"\n📊 INSIGHT: Monthly Charges Analysis")
    print(f"   - Avg Monthly Charges (Churned): ${avg_charges_churn:.2f}")
    print(f"   - Avg Monthly Charges (Not Churned): ${avg_charges_no_churn:.2f}")
    print(f"   - Higher charges correlate with higher churn!")


def plot_churn_by_services(df, save=True):
    """
    Analyze churn patterns based on services subscribed.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Filter to only columns that exist
    service_cols = [col for col in service_cols if col in df.columns]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(service_cols):
        churn_rate = df.groupby(col)['Churn'].mean() * 100
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(churn_rate)))
        
        bars = axes[idx].bar(range(len(churn_rate)), churn_rate.values, color=colors, 
                            edgecolor='black', linewidth=1.2)
        axes[idx].set_xticks(range(len(churn_rate)))
        axes[idx].set_xticklabels(churn_rate.index, rotation=45, ha='right', fontsize=9)
        axes[idx].set_title(f'Churn Rate by {col}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Churn Rate (%)', fontsize=10)
        axes[idx].set_ylim(0, 50)
        
        # Add value labels
        for bar, val in zip(bars, churn_rate.values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                          f'{val:.0f}%', ha='center', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(service_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '05_churn_by_services.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 05_churn_by_services.png")
    
    plt.close()
    
    print(f"\n📊 INSIGHT: Services Analysis")
    print(f"   - Customers WITHOUT security/backup/tech support churn more!")
    print(f"   - Fiber optic internet users show higher churn rates")
    print(f"   - Add-on services help with customer retention")


def plot_correlation_heatmap(df, save=True):
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): Dataset (should be encoded)
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    # Select numerical columns only
    numerical_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    correlation = numerical_df.corr()
    
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlGn_r', center=0, linewidths=0.5,
                annot_kws={'size': 8})
    
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '06_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 06_correlation_heatmap.png")
    
    plt.close()
    
    # Find top correlations with Churn
    if 'Churn' in correlation.columns:
        churn_corr = correlation['Churn'].drop('Churn').abs().sort_values(ascending=False)
        print(f"\n📊 INSIGHT: Correlation with Churn")
        print("   Top 5 features correlated with churn:")
        for feat, corr in churn_corr.head().items():
            print(f"   - {feat}: {corr:.3f}")


def plot_payment_method_analysis(df, save=True):
    """
    Analyze churn by payment method.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    ensure_viz_directory()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Churn rate by payment method
    payment_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
    payment_churn = payment_churn.sort_values(ascending=True)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(payment_churn)))
    bars = axes[0].barh(payment_churn.index, payment_churn.values, color=colors, 
                        edgecolor='black', linewidth=1.2)
    axes[0].set_title('Churn Rate by Payment Method', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Churn Rate (%)', fontsize=12)
    
    # Add value labels
    for bar, val in zip(bars, payment_churn.values):
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', fontsize=10)
    
    # Customer count by payment method
    payment_counts = df['PaymentMethod'].value_counts()
    axes[1].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(payment_counts))))
    axes[1].set_title('Payment Method Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(VIZ_PATH, '07_payment_method_analysis.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: 07_payment_method_analysis.png")
    
    plt.close()
    
    print(f"\n📊 INSIGHT: Payment Method Analysis")
    print(f"   - Electronic check users have the HIGHEST churn rate!")
    print(f"   - Automatic payments (bank/credit) have lower churn")
    print(f"   - Encourage customers to switch to automatic payments")


def run_full_eda(df, save=True):
    """
    Run complete EDA pipeline and generate all visualizations.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        save (bool): Whether to save plots
        
    Returns:
        dict: Summary of insights
    """
    print("=" * 60)
    print("🔍 EXPLORATORY DATA ANALYSIS - CUSTOMER CHURN")
    print("=" * 60)
    
    # Basic dataset info
    print(f"\n📁 Dataset Overview:")
    print(f"   - Total Samples: {len(df)}")
    print(f"   - Features: {len(df.columns) - 1}")
    print(f"   - Target: Churn (Binary)")
    
    # Generate all plots
    plot_churn_distribution(df, save)
    plot_churn_by_tenure(df, save)
    plot_churn_by_contract(df, save)
    plot_churn_by_charges(df, save)
    plot_churn_by_services(df, save)
    plot_payment_method_analysis(df, save)
    
    # For correlation, we need encoded data
    from feature_engineering import encode_categorical_features
    df_encoded, _ = encode_categorical_features(df, method='label')
    plot_correlation_heatmap(df_encoded, save)
    
    print("\n" + "=" * 60)
    print("✅ EDA Complete! All visualizations saved to 'visualizations/' folder")
    print("=" * 60)
    
    insights = {
        'churn_rate': (df['Churn'].mean() * 100),
        'avg_tenure_churn': df[df['Churn'] == 1]['tenure'].mean(),
        'avg_tenure_no_churn': df[df['Churn'] == 0]['tenure'].mean(),
        'avg_charges_churn': df[df['Churn'] == 1]['MonthlyCharges'].mean(),
        'avg_charges_no_churn': df[df['Churn'] == 0]['MonthlyCharges'].mean(),
    }
    
    return insights


if __name__ == "__main__":
    from data_preprocessing import load_and_clean_data
    
    print("Loading data for EDA...")
    df = load_and_clean_data()
    
    insights = run_full_eda(df)
    print(f"\n📈 Summary Insights: {insights}")
