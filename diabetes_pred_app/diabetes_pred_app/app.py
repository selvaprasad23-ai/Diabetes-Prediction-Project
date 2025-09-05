import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import ensure_dir, load_joblib
from data_loader import fetch_and_cache_dataset, get_feature_target
from train_ml import train_ml_models
from train_dl import train_dl_model

st.set_page_config(page_title="Diabetes Prediction (ML + DL)", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.caption("Compare Machine Learning and Deep Learning predictions on the Pima Indians Diabetes dataset.")

# Sidebar: actions
with st.sidebar:
    st.header("⚙️ Actions")
    if st.button("Fetch/Refresh Dataset"):
        df = fetch_and_cache_dataset()
        st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Cached to data/.")
    if st.button("Train/Re-train ML Models"):
        with st.spinner("Training ML models (RF + LogisticRegression)..."):
            metrics = train_ml_models()
        st.success(f"Done. RF AUC: {metrics['random_forest']['roc_auc']:.3f}, LR AUC: {metrics['log_reg_scaled']['roc_auc']:.3f}")
    if st.button("Train/Re-train DL Model"):
        with st.spinner("Training DL model (Keras Dense)... this can take a minute"):
            metrics = train_dl_model()
        m = metrics['dl_dense']
        st.success(f"Done. DL AUC: {m['roc_auc']:.3f}, ACC: {m['accuracy']:.3f}")
    st.divider()
    st.write("📁 Models are saved under `models/`.")

# Load dataset for UI
df = fetch_and_cache_dataset()
feature_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Input form
st.subheader("Enter Patient Values")
with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1, step=1)
        Glucose = st.number_input("Glucose", 0, 250, 120, step=1)
        BloodPressure = st.number_input("BloodPressure", 0, 200, 70, step=1)
        SkinThickness = st.number_input("SkinThickness", 0, 110, 20, step=1)
    with c2:
        Insulin = st.number_input("Insulin", 0, 1000, 79, step=1)
        BMI = st.number_input("BMI", 0.0, 80.0, 28.7, step=0.1)
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5, step=0.01, format="%.2f")
        Age = st.number_input("Age", 18, 100, 33, step=1)
    threshold = st.slider("Decision Threshold (probability ≥ threshold → Positive)", 0.1, 0.9, 0.5, 0.01)
    submitted = st.form_submit_button("Predict")

# Helper to load models (if exist)
def load_model_if_exists(path):
    return load_joblib(path) if os.path.exists(path) else None

rf_path = os.path.join("models", "random_forest.joblib")
lr_path = os.path.join("models", "log_reg_scaled.joblib")
dl_path = os.path.join("models", "keras_dense.keras")
dl_scaler_path = os.path.join("models", "dl_scaler.joblib")

rf_model = load_model_if_exists(rf_path)
lr_model = load_model_if_exists(lr_path)
dl_model = None
dl_scaler = None
if os.path.exists(dl_path) and os.path.exists(dl_scaler_path):
    try:
        from tensorflow import keras
        dl_model = keras.models.load_model(dl_path)
        dl_scaler = joblib.load(dl_scaler_path)
    except Exception as e:
        st.warning(f"Could not load DL model: {e}")

# Convert inputs to array
row = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

if submitted:
    # Ensure at least ML models exist
    if rf_model is None or lr_model is None:
        with st.spinner("Training ML models for the first time..."):
            _ = train_ml_models()
            rf_model = load_joblib(rf_path)
            lr_model = load_joblib(lr_path)

    results = []
    # RF
    rf_proba = rf_model.predict_proba(row)[0,1]
    results.append(("Random Forest", rf_proba, int(rf_proba >= threshold)))

    # LR
    lr_proba = lr_model.predict_proba(row)[0,1]
    results.append(("Logistic Regression (scaled)", lr_proba, int(lr_proba >= threshold)))

    # DL (optional)
    if dl_model is not None and dl_scaler is not None:
        row_s = dl_scaler.transform(row)
        dl_proba = float(dl_model.predict(row_s, verbose=0).ravel()[0])
        results.append(("Deep Learning (Keras Dense)", dl_proba, int(dl_proba >= threshold)))
    else:
        results.append(("Deep Learning (Keras Dense)", np.nan, None))

    # Show results
    st.subheader("Results")
    for name, proba, pred in results:
        if np.isnan(proba):
            st.info(f"{name}: not available (train the DL model from the sidebar if you want this).")
            continue
        st.write(f"**{name}** → Probability: **{proba:.3f}** → Prediction: **{'Positive' if pred==1 else 'Negative'}**")

    st.caption("These are model-based estimates; consult a clinician for medical decisions.")

# Show simple dataset preview & class balance
with st.expander("Dataset Preview & Class Balance"):
    st.dataframe(df.head(10))
    pos_rate = df['Outcome'].mean()
    st.write(f"Positive rate in dataset: **{pos_rate:.3f}**")
