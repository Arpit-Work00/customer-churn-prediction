# 🎯 Customer Churn Prediction System

A complete end-to-end Machine Learning project to predict customer churn in the telecom industry. Built to demonstrate full-stack ML skills including data processing, model training, and web deployment.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-pre-5x38sey3betrfbzexjq4ii.streamlit.app/)

**👉 [Try the Live Demo](https://customer-churn-pre-5x38sey3betrfbzexjq4ii.streamlit.app/)**

---

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)

## 🎯 Project Overview

Customer churn is a critical business metric that measures the rate at which customers stop doing business with a company. In the telecom industry, acquiring new customers is 5-25x more expensive than retaining existing ones. This project develops an AI-powered system to:

- **Predict** which customers are likely to churn
- **Analyze** key factors contributing to churn
- **Recommend** retention strategies

## 📌 Problem Statement

### What is Customer Churn?
Customer churn (also known as customer attrition) refers to when customers stop using a company's product or service. For telecom companies, this means customers canceling their subscriptions.

### Business Impact
- A 5% increase in customer retention can boost profits by 25-95%
- Churned customers may share negative experiences
- Proactive intervention can save at-risk customers

### Objective
Develop a machine learning model that predicts whether a customer will churn based on their demographics, services, and account information.

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Framework** | Streamlit |
| **Model Persistence** | Joblib |
| **Class Imbalance** | imbalanced-learn (SMOTE) |

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   └── telco_customer_churn.csv    # Dataset
├── models/
│   └── best_model.pkl              # Trained model
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning functions
│   ├── feature_engineering.py      # Feature encoding & scaling
│   ├── eda_analysis.py             # Exploratory data analysis
│   ├── model_training.py           # Model training pipeline
│   └── utils.py                    # Utility functions
├── visualizations/                  # Generated plots
├── app.py                          # Streamlit web application
├── requirements.txt
├── README.md
└── INTERVIEW_PREP.md               # Interview Q&A
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃 How to Run

### Option 1: Run the Web Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

### Option 2: Run Model Training Separately
```bash
cd src
python model_training.py
```

### Option 3: Run EDA Analysis
```bash
cd src
python eda_analysis.py
```

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset, which contains information about:

| Feature Category | Features |
|-----------------|----------|
| **Demographics** | Gender, Senior Citizen, Partner, Dependents |
| **Services** | Phone, Internet, Security, Backup, Support, Streaming |
| **Account** | Contract type, Payment method, Tenure, Charges |
| **Target** | Churn (Yes/No) |

**Dataset Statistics:**
- **Total Samples:** 7,043 customers
- **Features:** 20
- **Target:** Binary (Churn: Yes/No)
- **Churn Rate:** ~26.5%

## 📈 Model Performance

Four machine learning models were trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80 | 0.66 | 0.52 | 0.58 | 0.84 |
| Decision Tree | 0.78 | 0.57 | 0.51 | 0.54 | 0.71 |
| Random Forest | 0.82 | 0.67 | 0.53 | 0.59 | 0.85 |
| **XGBoost** | **0.83** | **0.68** | **0.55** | **0.61** | **0.86** |

**Best Model:** XGBoost with 86% ROC-AUC score

### Key Findings
1. **Contract Type** is the strongest predictor of churn
2. **Month-to-month** contracts have 3x higher churn rate
3. **Electronic check** payment method correlates with higher churn
4. Customers without **security/support services** churn more

## 🔮 Future Improvements

1. **Deep Learning**: Implement neural network for comparison
2. **Real-time Data**: Connect to actual customer database
3. **A/B Testing**: Test retention strategies
4. **API Deployment**: Deploy as REST API
5. **Mobile App**: Build mobile interface

## ⚠️ Limitations

- Model trained on historical data only
- Assumes features are available at prediction time
- No real-time data integration
- Limited to telecom industry context

## 👨‍💻 Author

This project demonstrates end-to-end Machine Learning skills including data processing, model training, evaluation, and deployment.

⭐ **If you found this project helpful, please give it a star!** ⭐
