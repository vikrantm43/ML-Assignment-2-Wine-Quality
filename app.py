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
    
    # FIX: Standardize uploaded column names to match model training
    # This turns "fixed acidity" into "fixed_acidity" automatically
    test_df.columns = [c.strip().replace(' ', '_').lower() for c in test_df.columns]
    
    st.write("### Data Preview")
    st.dataframe(test_df.head(3))

    # Validation
    if all(col in test_df.columns for col in feature_cols) and 'target' in test_df.columns:
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        # Scale
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        model = joblib.load(f"./models/{model_map[selected_model]}")
        if selected_model in ['Random Forest', 'XGBoost']:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc:.2%}")
        c2.metric("F1 Score", f"{f1:.3f}")

        # Plots
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='magma', ax=ax)
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred))
    else:
        st.error(f"**Error:** Missing columns. Required: {', '.join(feature_cols)} and **target**")
        st.write("Your columns:", list(test_df.columns))
else:
    st.info("Please upload a CSV file to begin. Ensure it has the 11 chemical features and a 'target' column.")

st.markdown("---")
st.caption("UCI Red Wine Quality Dataset Analysis")
