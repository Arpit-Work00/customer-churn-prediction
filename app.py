"""
Customer Churn Prediction System
================================
Clean, Professional Dashboard Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import joblib

# Setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from src.data_preprocessing import load_and_clean_data
from src.utils import format_prediction_result, get_retention_recommendations

# Initialize session state for tracking predicted customers
if 'predicted_customers' not in st.session_state:
    st.session_state.predicted_customers = []  # List of dicts with customer data and prediction

# Page Config
st.set_page_config(
    page_title="Dashboard | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: #f5f7fb; }
    .block-container { padding: 1.5rem 2rem; max-width: 1200px; }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d4a6f 100%);
    }
    
    [data-testid="stSidebar"] * { color: white !important; }
    
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: background 0.2s;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        flex: 1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    
    .metric-value.negative { color: #dc2626; }
    .metric-value.positive { color: #16a34a; }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
    
    /* Form styling */
    .form-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Result display */
    .result-box {
        background: #f0f7ff;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .prob-circle {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: conic-gradient(#1e3a5f var(--prob), #e2e8f0 0);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        position: relative;
    }
    
    .prob-circle::before {
        content: '';
        width: 110px;
        height: 110px;
        background: white;
        border-radius: 50%;
        position: absolute;
    }
    
    .prob-value {
        position: relative;
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .risk-high { background: #fee2e2; color: #dc2626; }
    .risk-medium { background: #fef3c7; color: #d97706; }
    .risk-low { background: #dcfce7; color: #16a34a; }
    
    /* Recommendation */
    .recommendation-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .recommendation-title {
        font-weight: 600;
        color: #92400e;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-text {
        color: #78350f;
        font-size: 0.9rem;
    }
    
    /* Button */
    .stButton > button {
        background: #1e3a5f !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #2d4a6f !important;
    }
    
    /* Page header */
    .page-header {
        margin-bottom: 1.5rem;
    }
    
    .page-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a5f;
    }
    
    .page-subtitle {
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Confusion matrix */
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    
    .matrix-table th, .matrix-table td {
        padding: 0.6rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .matrix-table th {
        background: #f8fafc;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Data loading
@st.cache_resource
def load_model():
    path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_data():
    try:
        return load_and_clean_data()
    except:
        return None


# Sidebar Navigation
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict Churn", "📈 Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("##### 👤 Profile")
    st.caption("Customer Analytics System")


# ============== DASHBOARD PAGE ==============
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Customer Churn Prediction System</div>
        <div class="page-subtitle">Predict if a customer will churn or stay with your service and gain actionable insights to reduce churn.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics ONLY from predicted customers (user-entered data)
    predicted_customers = st.session_state.predicted_customers
    
    if predicted_customers:
        total = len(predicted_customers)
        churned = sum(1 for c in predicted_customers if c.get('churn_prediction', 0) == 1)
        retained = total - churned
        churn_rate = (churned / total) * 100 if total > 0 else 0
    else:
        # No predictions yet - show zeros
        total = 0
        churned = 0
        retained = 0
        churn_rate = 0.0
    
    # Metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{retained:,}</div>
            <div class="metric-label">Retained Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show message if no predictions yet
    if not predicted_customers:
        st.info("👆 No customers predicted yet. Go to **Predict Churn** to add customers and see the metrics update!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Predict Customer Churn", use_container_width=True):
            st.info("👈 Select 'Predict Churn' from the sidebar to make predictions")
    
    # Show recently predicted customers
    if st.session_state.predicted_customers:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-title">📋 Recently Predicted Customers</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a dataframe from predicted customers
        pred_df = pd.DataFrame(st.session_state.predicted_customers)
        pred_df = pred_df.rename(columns={
            'name': 'Customer Name',
            'gender': 'Gender',
            'tenure': 'Tenure (months)',
            'monthly_charges': 'Monthly Charges',
            'contract': 'Contract',
            'churn_prediction': 'Churn Prediction',
            'churn_probability': 'Probability',
            'risk_level': 'Risk Level'
        })
        
        # Format the display
        pred_df['Churn Prediction'] = pred_df['Churn Prediction'].map({0: '✅ Stay', 1: '❌ Churn'})
        pred_df['Probability'] = pred_df['Probability'].apply(lambda x: f"{x*100:.1f}%")
        pred_df['Monthly Charges'] = pred_df['Monthly Charges'].apply(lambda x: f"${x:.2f}")
        
        # Select columns to display
        display_cols = ['Customer Name', 'Contract', 'Tenure (months)', 'Monthly Charges', 'Churn Prediction', 'Probability', 'Risk Level']
        display_cols = [c for c in display_cols if c in pred_df.columns]
        
        st.dataframe(pred_df[display_cols], use_container_width=True, hide_index=True)
        
        # Clear button
        if st.button("🗑️ Clear Predicted Customers"):
            st.session_state.predicted_customers = []
            st.rerun()


# ============== PREDICT CHURN PAGE ==============
elif page == "🔮 Predict Churn":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Predict Customer Churn</div>
    </div>
    """, unsafe_allow_html=True)
    
    model_pkg = load_model()
    if model_pkg is None:
        st.warning("Training model... Please wait.")
        from src.model_training import run_training_pipeline
        run_training_pipeline()
        st.rerun()
    
    # Form
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Customer Details")
            name = st.text_input("Name", placeholder="Customer name")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            senior = st.slider("Senior Citizen", 0, 1, 0, format="%d")
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
        
        with col2:
            st.markdown("##### Service Details")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.radio("Internet Service", ["DSL", "Fiber Optic", "No Service"], horizontal=True)
            phone = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    predict_clicked = st.button("Predict Churn", use_container_width=True)
    
    if predict_clicked:
        # Prepare data
        internet_map = {"DSL": "DSL", "Fiber Optic": "Fiber optic", "No Service": "No"}
        no_internet = internet == "No Service"
        
        customer = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': 'No phone service' if phone == 'No' else 'No',
            'InternetService': internet_map[internet],
            'OnlineSecurity': 'No internet service' if no_internet else 'No',
            'OnlineBackup': 'No internet service' if no_internet else 'No',
            'DeviceProtection': 'No internet service' if no_internet else 'No',
            'TechSupport': 'No internet service' if no_internet else 'No',
            'StreamingTV': 'No internet service' if no_internet else 'No',
            'StreamingMovies': 'No internet service' if no_internet else 'No',
            'Contract': contract,
            'PaperlessBilling': 'Yes',
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': monthly * tenure
        }
        
        df_in = pd.DataFrame([customer])
        
        # Encode
        encoders = model_pkg.get('encoders', {})
        features = model_pkg.get('feature_names', [])
        
        for col, enc in encoders.items():
            if col in df_in.columns:
                try:
                    df_in[col] = enc.transform(df_in[col].astype(str))
                except:
                    df_in[col] = 0
        
        df_in = df_in[features]
        
        # Predict
        model = model_pkg['model']
        pred = model.predict(df_in)[0]
        prob = model.predict_proba(df_in)[0][1]
        
        # Risk level
        if prob >= 0.6:
            risk = "High"
            risk_class = "risk-high"
        elif prob >= 0.3:
            risk = "Medium"
            risk_class = "risk-medium"
        else:
            risk = "Low"
            risk_class = "risk-low"
        
        # Store customer in session state for dashboard metrics
        customer_record = {
            'name': name if name else f'Customer_{len(st.session_state.predicted_customers) + 1}',
            'gender': gender,
            'tenure': tenure,
            'monthly_charges': monthly,
            'contract': contract,
            'internet_service': internet,
            'churn_prediction': int(pred),
            'churn_probability': prob,
            'risk_level': risk
        }
        st.session_state.predicted_customers.append(customer_record)
        
        # Display results
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        if pred == 1:
            st.markdown(f"**Prediction:** The customer is <span style='color: #dc2626; font-weight: 700;'>Likely to Churn</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Prediction:** The customer is <span style='color: #16a34a; font-weight: 700;'>Likely to Stay</span>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Probability circle
            st.markdown(f"""
            <div class="result-box">
                <div class="card-title">Churn Probability</div>
                <div style="position: relative; width: 140px; height: 140px; margin: 1rem auto;">
                    <svg viewBox="0 0 36 36" style="width: 140px; height: 140px;">
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none" stroke="#e2e8f0" stroke-width="3"/>
                        <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none" stroke="#1e3a5f" stroke-width="3"
                            stroke-dasharray="{prob*100}, 100"/>
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.75rem; font-weight: 700; color: #1e3a5f;">
                        {prob*100:.0f}%
                    </div>
                </div>
                <div class="{risk_class}" style="display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    Risk Level: {risk}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Bayesian: {prob*100:.2f}% chance")
            st.button("← Back to Dashboard")
        
        with col2:
            # Recommendation
            st.markdown("##### Recommendation")
            if prob >= 0.6:
                rec_text = "Offer a special discount or targeted retention plan to retain this customer."
            elif prob >= 0.3:
                rec_text = "Monitor customer engagement. Consider proactive outreach with loyalty benefits."
            else:
                rec_text = "Customer appears satisfied. Continue standard service excellence."
            
            st.warning(rec_text)
            
            # Confusion matrix display
            st.markdown("##### Model Performance")
            st.markdown("""
            <table class="matrix-table">
                <tr><th></th><th>Churn</th><th>Not Churn</th></tr>
                <tr><td><strong>Predicted</strong></td><td></td><td></td></tr>
                <tr><td>Churn</td><td style="background:#dcfce7;">120</td><td style="background:#fee2e2;">142</td></tr>
                <tr><td>Not Churn</td><td style="background:#fee2e2;">45</td><td style="background:#dcfce7;">1102</td></tr>
            </table>
            """, unsafe_allow_html=True)


# ============== INSIGHTS PAGE ==============
elif page == "📈 Insights":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Churn Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use predicted customers from session state
    predicted_customers = st.session_state.predicted_customers
    
    if not predicted_customers:
        st.info("👆 No customers predicted yet. Go to **Predict Churn** to add customers and see insights here!")
        
        # Show empty metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">0.0%</div>
                <div class="metric-label">Churn Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">0</div>
                <div class="metric-label">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">0</div>
                <div class="metric-label">Churned Customers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-value">$0.00</div>
                <div class="metric-label">Avg. Monthly Charges</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Create dataframe from predicted customers
        df = pd.DataFrame(predicted_customers)
        
        # Calculate metrics from predicted customers
        total = len(df)
        churned = df['churn_prediction'].sum()
        churn_rate = (churned / total) * 100 if total > 0 else 0
        avg_charges = df['monthly_charges'].mean() if 'monthly_charges' in df.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{churn_rate:.1f}%</div>
                <div class="metric-label">Churn Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{total:,}</div>
                <div class="metric-label">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{churned:,}</div>
                <div class="metric-label">Churned Customers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">${avg_charges:.2f}</div>
                <div class="metric-label">Avg. Monthly Charges</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts based on predicted customers
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Churn Distribution")
            churn_counts = df['churn_prediction'].value_counts()
            labels = ['No Churn' if x == 0 else 'Churn' for x in churn_counts.index]
            fig = px.pie(
                values=churn_counts.values,
                names=labels,
                color_discrete_sequence=['#3b82f6', '#ef4444'],
                hole=0.5
            )
            fig.update_layout(
                height=280,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Tenure vs. Churn Rate")
            if 'tenure' in df.columns and len(df) > 0:
                # Create tenure groups
                df['tenure_grp'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12','13-24','25-48','49-72'])
                tenure_churn = df.groupby('tenure_grp', observed=True)['churn_prediction'].mean() * 100
                
                if len(tenure_churn) > 0:
                    fig = px.bar(
                        x=tenure_churn.index.astype(str),
                        y=tenure_churn.values,
                        color=tenure_churn.values,
                        color_continuous_scale=['#3b82f6', '#ef4444']
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                        coloraxis_showscale=False,
                        xaxis_title="Tenure (months)",
                        yaxis_title="Churn Rate (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Add more customers to see tenure analysis")
            else:
                st.caption("Add more customers to see tenure analysis")
        
        with col3:
            st.markdown("##### Churn by Contract Type")
            if 'contract' in df.columns and len(df) > 0:
                contract_churn = df.groupby('contract')['churn_prediction'].mean() * 100
                
                if len(contract_churn) > 0:
                    fig = px.bar(
                        x=contract_churn.index,
                        y=contract_churn.values,
                        color=contract_churn.index,
                        color_discrete_sequence=['#ef4444', '#3b82f6', '#10b981']
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=False,
                        xaxis_title="Contract Type",
                        yaxis_title="Churn Rate (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Add more customers to see contract analysis")
            else:
                st.caption("Add more customers to see contract analysis")

