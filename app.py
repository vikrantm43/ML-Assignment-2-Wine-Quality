import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model



st.title("🍷 Wine Quality Classification")

@st.cache_resource
def prepare_models():
    # Data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(url, sep=';')
    data['target'] = (data['quality'] >= 6).astype(int)
    X = data.drop(['quality', 'target'], axis=1).values  # NUMPY - no names!
    y = data['target'].values
    
    X_train, X_test, y_train, y_test = np.split(X, [int(0.8*len(X))], axis=0)  # Simple split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('./models', exist_ok=True)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train),
        'Decision Tree': DecisionTreeClassifier().fit(X_train_scaled, y_train),
        'KNN': KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train),
        'Naive Bayes': GaussianNB().fit(X_train_scaled, y_train),
        'Random Forest': RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss').fit(X_train, y_train)
    }
    
    joblib.dump(scaler, './models/scaler.pkl')
    model_map = {}
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, f'./models/{filename}')
        model_map[name] = filename
    
    return model_map, scaler

model_map, scaler = prepare_models()

st.sidebar.header("📁 Upload Test CSV")
uploaded_file = st.sidebar.file_uploader("Choose CSV", type="csv")

model_names = list(model_map.keys())
selected_model = st.sidebar.selectbox("Choose Model", model_names)

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("**Preview:**", test_df.head(3))
    
    required_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                     'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                     'pH', 'sulphates', 'alcohol']
    
    if all(col in test_df.columns for col in required_cols) and 'target' in test_df.columns:
        X_test = test_df[required_cols].values  # NUMPY!
        y_test = test_df['target'].values
        
        X_test_scaled = scaler.transform(X_test)
        
        model_filename = model_map[selected_model]
        model = joblib.load(f"./models/{model_filename}")
        
        # RF/XGB use unscaled; others scaled
        if selected_model in ['Random Forest', 'XGBoost']:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("F1 Score", f"{f1:.3f}")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))
        
    else:
        st.error("**CSV must have exact columns:** fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,**target**")
        st.info("Use sample CSV from previous messages.")
else:
    st.info("👆 Upload CSV with 11 features + 'target' column")

st.markdown("---")
st.caption("Trained on UCI Wine Quality dataset. Deadline-proof!")
