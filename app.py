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
    # Load original training data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    
    # Standardize training column names
    data.columns = [c.replace(' ', '_') for c in data.columns]
    data['target'] = (data['quality'] >= 6).astype(int)
    
    feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol']
    
    X = data[feature_cols].values
    y = data['target'].values
    
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    os.makedirs('./models', exist_ok=True)
    
    # Train models
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
    # 1. Load User Data
    test_df = pd.read_csv(uploaded_file)
    original_display_df = test_df.copy() # Keep for the final table
    
    # 2. Standardize names to lowercase/underscores to handle variations
    test_df.columns = [c.strip().replace(' ', '_').lower() for c in test_df.columns]
    
    # 3. Apply Column Mapping (Fixes your 'citric_acidity' and 'ph' issue)
    mapping = {
        'citric_acidity': 'citric_acid',
        'ph': 'pH'
    }
    test_df = test_df.rename(columns=mapping)
    
    # Define columns model expects
    required_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                     'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                     'pH', 'sulphates', 'alcohol']
    
    # 4. Process and Predict
    if all(col in test_df.columns for col in required_cols) and 'target' in test_df.columns:
        X_test = test_df[required_cols].values
        y_test = test_df['target'].values
        
        # Scale for Linear/Distance models (Logic matches your requirement)
        X_test_scaled = scaler.transform(X_test)
        
        # Load the selected model
        model = joblib.load(f"./models/{model_map[selected_model]}")
        
        if selected_model in ['Random Forest', 'XGBoost']:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
        
        # --- OUTPUT SECTION ---
        st.success(f"Successfully analyzed {len(test_df)} wine samples using {selected_model}!")

        # A. Detailed Results Table
        st.subheader("📋 Prediction Results")
        results_df = original_display_df.copy()
        results_df['Predicted_Quality'] = ["Good (1)" if p == 1 else "Bad (0)" for p in y_pred]
        st.dataframe(results_df.style.highlight_max(subset=['Predicted_Quality'], color='#d4edda'))

        # B. Metrics
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        col2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")
        
        # C. Confusion Matrix Visualization
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

    else:
        st.error("Missing columns! Please ensure your CSV has the 11 chemical features and 'target'.")
        st.write("Found columns:", list(test_df.columns))
else:
    st.info("👋 Upload your `data.csv` in the sidebar to get started.")

st.markdown("---")
st.caption("UCI Red Wine Quality Classification System")
