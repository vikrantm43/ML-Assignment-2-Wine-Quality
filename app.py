import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷")

# --- Model Preparation (Cached) ---
@st.cache_resource
def prepare_models():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    
    # Standardize column names for training: replace spaces with underscores
    data.columns = [c.replace(' ', '_') for c in data.columns]
    
    data['target'] = (data['quality'] >= 6).astype(int)
    # Features (11 columns)
    feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol']
    
    X = data[feature_cols].values
    y = data['target'].values
    
    # Split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    os.makedirs('./models', exist_ok=True)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train),
        'Random Forest': RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss').fit(X_train, y_train),
        'Decision Tree': DecisionTreeClassifier().fit(X_train_scaled, y_train),
        'KNN': KNeighborsClassifier().fit(X_train_scaled, y_train)
    }
    
    model_map = {}
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, f'./models/{filename}')
        model_map[name] = filename
        
    return model_map, scaler, feature_cols

model_map, scaler, feature_cols = prepare_models()

# --- UI Layout ---
st.title("🍷 Wine Quality Classification")

st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")
selected_model = st.sidebar.selectbox("Select Model", list(model_map.keys()))

if uploaded_file:
    # Read CSV
    test_df = pd.read_csv(uploaded_file)
    
    # 1. Clean up column names (lowercase and underscores)
    test_df.columns = [c.strip().replace(' ', '_').lower() for c in test_df.columns]
    
    # 2. FIX: Map the variations in your file to what the model expects
    column_mapping = {
        'citric_acidity': 'citric_acid',
        'ph': 'pH'  # Ensure it matches the "Required" list exactly
    }
    test_df = test_df.rename(columns=column_mapping)
    
    st.write("### Data Preview")
    st.dataframe(test_df.head(3))

    # Define the exact required list (must match what's in the error)
    feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol']

    # Validation
    if all(col in test_df.columns for col in feature_cols) and 'target' in test_df.columns:
        # ... (rest of your prediction logic)
        X_test = test_df[feature_cols].values
        # ...
