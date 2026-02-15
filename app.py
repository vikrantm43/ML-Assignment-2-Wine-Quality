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
    
    # Standardize training column names (fixed acidity -> fixed_acidity)
    data.columns = [c.replace(' ', '_') for c in data.columns]
    
    # Define Target: 1 for Good (>=6), 0 for Bad
    data['target'] = (data['quality'] >= 6).astype(int)
    
    feature_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                    'pH', 'sulphates', 'alcohol']
    
    X = data[feature_cols].values
    y = data['target'].values
    
    # Simple Train/Test Split for the demo
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    os.makedirs('./models', exist_ok=True)
    
    # Train a suite of models
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
    # 1. Load User Data with auto-delimiter detection
    try:
        test_df = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    # 2. Standardize column names (fix spaces and quotes)
    test_df.columns = [c.strip().replace('"', '').replace(' ', '_') for c in test_df.columns]
    
    # 3. Create 'target' if only 'quality' exists
    if 'quality' in test_df.columns and 'target' not in test_df.columns:
        test_df['target'] = (test_df['quality'] >= 6).astype(int)
    
    # 4. Handle specific naming inconsistencies
    mapping = {
        'citric_acidity': 'citric_acid',
        'ph': 'pH' # Ensure casing matches model expectations
    }
    test_df = test_df.rename(columns=mapping)
    
    st.write("### Data Preview")
    st.dataframe(test_df.head(5))
    
    # 5. Process and Predict
    if all(col in test_df.columns for col in feature_cols) and 'target' in test_df.columns:
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        # Scaling logic
        X_test_scaled = scaler.transform(X_test)
        
        model = joblib.load(f"./models/{model_map[selected_model]}")
        
        if selected_model in ['Random Forest', 'XGBoost']:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
        
        # --- OUTPUTS ---
        st.success(f"Analysis complete for {len(test_df)} samples!")
        
        # Results Table
        results_df = test_df.copy()
        results_df['Prediction'] = ["Good (1)" if p == 1 else "Bad (0)" for p in y_pred]
        st.subheader("📋 Classification Table")
        st.dataframe(results_df[['target', 'Prediction'] + feature_cols])

        # Metrics
        st.subheader("📊 Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        c2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='RdPu')
        plt.title(f"Confusion Matrix: {selected_model}")
        st.pyplot(fig)
    else:
        st.error("Missing required columns!")
        st.write("Model needs: `fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `pH`, `sulphates`, `alcohol`, and `target` (or `quality`).")
        st.write("Detected in your file:", list(test_df.columns))

else:
    st.info("👋 Upload `winequality-red.csv` in the sidebar to view the results.")

st.markdown("---")
st.caption("UCI Wine Quality Dataset Analysis • Streamlit App")
