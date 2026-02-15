!pip install streamlit
import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# App Title [cite: 9]
st.title("🍷 Wine Quality Classification Dashboard")

# Step a: Dataset upload option (CSV) [cite: 91]
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

# Step b: Model selection dropdown [cite: 92]
st.sidebar.header("2. Choose Model")
model_option = st.sidebar.selectbox(
    "Select a Classification Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)

def load_model(name):
    # Mapping selection to saved filenames
    model_map = {
        "Logistic Regression": "lr_model.pkl",
        "Decision Tree": "dt_model.pkl",
        "KNN": "knn_model.pkl",
        "Naive Bayes": "nb_model.pkl",
        "Random Forest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl"
    }
    with open(os.path.join('model', model_map[name]), 'rb') as f:
        return pickle.load(f)

if uploaded_file is not None:
    # Load test data [cite: 91]
    test_df = pd.read_csv(uploaded_file)
    st.write("### Test Data Preview")
    st.dataframe(test_df.head())

    # Pre-processing (Assuming target column 'target' is present)
    if 'target' in test_df.columns:
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Load selected model
        model = load_model(model_option)
        y_pred = model.predict(X_test)

        # Step c: Display of evaluation metrics [cite: 93]
        st.write(f"## Evaluation Metrics: {model_option}")
        col1, col2, col3 = st.columns(3)

        # Note: You can calculate these dynamically or display pre-saved values [cite: 40-45]
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("F1 Score", f"{f1:.2f}")

        # Step d: Confusion matrix or classification report [cite: 94]
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        st.error("The uploaded CSV must contain a 'target' column for evaluation.")
else:
    st.info("Please upload a test CSV file to begin.")
