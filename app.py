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
st.title("🍷 Wine Quality Classification")

@st.cache_resource
def prepare_models():
    # Data loading
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    
    # Preprocessing: 1 for Good (>=6), 0 for Bad
    data['target'] = (data['quality'] >= 6).astype(int)
    X = data.drop(['quality', 'target'], axis=1).values
    y = data['target'].values
    
    # Train/Test Split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    os.makedirs('./models', exist_ok=True)
    
    # Training suite
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train),
        'Decision Tree': DecisionTreeClassifier().fit(X_train_scaled, y_train),
        'KNN': KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train),
        'Naive Bayes': GaussianNB().fit(X_train_scaled, y_train),
        'Random Forest': RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss').fit(X_train, y_train)
    }
    
    # Save artifacts
    joblib.dump(scaler, './models/scaler.pkl')
    model_map = {}
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, f'./models/{filename}')
        model_map[name] = filename
    
    return model_map, scaler, data.head(5)

model_map, scaler, sample_data = prepare_models()

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
model_names = list(model_map.keys())
selected_model = st.sidebar.selectbox("Choose Model", model_names)

st.sidebar.markdown("---")
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")

# Helper: Download sample template
sample_csv = sample_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Sample CSV", sample_csv, "wine_sample.csv", "text/csv")

# --- Main Logic ---
required_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                 'pH', 'sulphates', 'alcohol']

if uploaded_file:
    # Handle both semicolon and comma separated files
    try:
        test_df = pd.read_csv(uploaded_file)
    except:
        test_df = pd.read_csv(uploaded_file, sep=';')
        
    st.write("### Preview of Uploaded Data")
    st.dataframe(test_df.head(5))
    
    # Normalize column names (replace spaces with underscores if necessary)
    test_df.columns = [c.replace(' ', '_') for c in test_df.columns]
    check_cols = [c.replace(' ', '_') for c in required_cols]
    
    if all(col in test_df.columns for col in check_cols) and 'target' in test_df.columns:
        X_test = test_df[check_cols].values
        y_test = test_df['target'].values
        
        # Scale for specific models
        X_test_scaled = scaler.transform(X_test)
        
        model_filename = model_map[selected_model]
        model = joblib.load(f"./models/{model_filename}")
        
        # Prediction logic
        if selected_model in ['Random Forest', 'XGBoost']:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
        
        # Metrics Display
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("F1 Score", f"{f1:.3f}")
        
        # Visuals
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='RdPu', ax=ax)
            st.pyplot(fig)
            
        with col_right:
            st.subheader("Classification Report")
            st.code(classification_report(y_test, y_pred))
            
    else:
        st.error(f"**Column mismatch!** CSV needs: {', '.join(check_cols)} and **target**")
else:
    st.info("💡 **Getting started:** Download the sample CSV from the sidebar and upload it here to see the model in action.")

st.markdown("---")
st.caption("Built with Streamlit • Model: " + selected_model)
