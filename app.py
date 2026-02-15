import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ONE-TIME: Train & save models (cached, fast)
@st.cache_resource
def train_and_save_models():
    # Load data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    data['target'] = (data['quality'] >= 6).astype(int)
    X = data.drop(['quality', 'target'], axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
        'Decision Tree': DecisionTreeClassifier(random_state=42).fit(X_train_scaled, y_train),
        'KNN': KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train),
        'Naive Bayes': GaussianNB().fit(X_train_scaled, y_train),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss').fit(X_train, y_train)
    }
    
    # Save
    joblib.dump(scaler, 'models/scaler.pkl')
    for name, model in models.items():
        safename = name.lower().replace(' ', '').replace('-', '')
        joblib.dump(model, f'models/{safename}.pkl')
    
    return models, scaler

# Train once
models_dict, scaler = train_and_save_models()

# Rest of app (model_map, load_model, UI) – copy from previous app.py lines 42-end
st.title("Wine Quality Classification Dashboard")

st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type="csv")

st.sidebar.header("2. Choose Model")
model_option = st.sidebar.selectbox("Select Model", list(models_dict.keys()))

model_map = {name: name.lower().replace(' ', '').replace('-', '') for name in models_dict.keys()}

@st.cache_resource
def load_model_cached(name):
    safename = model_map[name]
    return joblib.load(f"./models/{safename}.pkl")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("Test Data Preview:", test_df.head())
    
    if 'target' in test_df.columns:
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        X_test_scaled = scaler.transform(X_test)
        
        model = load_model_cached(model_option)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.subheader(f"{model_option} Results")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)
        
        st.code(classification_report(y_test, y_pred))
    else:
        st.error("CSV needs 'target' column (0/1)")
else:
    st.info("👆 Upload CSV first!")
