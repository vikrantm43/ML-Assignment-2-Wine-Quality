import streamlit as st
import pandas as pd
import joblib  # Use joblib for all (consistent with your saves)
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Wine Quality Classification Dashboard")

# Sidebar
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type="csv")

st.sidebar.header("2. Choose Model")
model_option = st.sidebar.selectbox(
    "Select a Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Model mapping (filenames without .pkl)
model_map = {
    "Logistic Regression": "lrmodel",
    "Decision Tree": "dtmodel",
    "KNN": "knnmodel",
    "Naive Bayes": "nbmodel",
    "Random Forest": "rfmodel",
    "XGBoost": "xgbmodel"
}

# Load scaler (shared for scaled models)
@st.cache_resource  # Cache for performance
def load_scaler():
    return joblib.load("./models/scaler.pkl")

# Load model function
@st.cache_resource
def load_model(model_name):
    model_filename = model_map[model_name] + ".pkl"
    model_path = os.path.join("./models", model_filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file {model_path} not found!")
        return None

if uploaded_file is not None:
    # Load and preview test data
    test_df = pd.read_csv(uploaded_file)
    st.write("Test Data Preview")
    st.dataframe(test_df.head())

    # Assume 'target' column exists (binary: 0/1)
    if 'target' in test_df.columns:
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Load scaler and scale (for LR, DT(scaled), KNN, NB)
        scaler = load_scaler()
        X_test_scaled = scaler.transform(X_test)

        # Load selected model
        model = load_model(model_option)
        if model:
            # Predict (use scaled for models that need it; RF/XGB don't but safe)
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.write(f"**Evaluation Metrics: {model_option}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("F1 Score", f"{f1:.4f}")

            # Confusion Matrix
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{model_option} Confusion Matrix")
            st.pyplot(fig)

            # Classification Report
            st.write("**Classification Report**")
            st.text(classification_report(y_test, y_pred))
    else:
        st.error("The uploaded CSV must contain a 'target' column for evaluation.")
else:
    st.info("Please upload a test CSV file to begin.")
