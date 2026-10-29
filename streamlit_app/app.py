"""
Streamlit demo interface for LUNARA-AI models.

Usage:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.predictor import ModelPredictor

st.set_page_config(page_title="LUNARA-AI", page_icon="🌙", layout="wide")

st.title("LUNARA: Multi-Model Pregnancy Health Prediction")
st.markdown(
    "Enter patient details to predict risks for Gestational Diabetes, "
    "Maternal Health, or Preeclampsia."
)

predictor = ModelPredictor(PROJECT_ROOT / "ml")

tab1, tab2, tab3 = st.tabs([
    "Gestational Diabetes",
    "Maternal Health Risk",
    "Preeclampsia Risk",
])

with tab1:
    st.header("Gestational Diabetes Prediction")
    with st.form("gdm_form"):
        age = st.number_input("Age", min_value=18, max_value=50, value=30)
        no_pregnancy = st.number_input("Number of Pregnancies", min_value=0, max_value=10, value=2)
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=28.0)
        hdl = st.number_input("HDL (mg/dL)", min_value=20.0, max_value=100.0, value=50.0)
        family_history = st.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        large_child = st.selectbox("Large Child or Birth Defect", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        pcos = st.selectbox("PCOS", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sys_bp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=200.0, value=120.0)
        dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=80.0)
        ogtt = st.number_input("OGTT (mg/dL)", min_value=50.0, max_value=300.0, value=130.0)
        sedentary = st.selectbox("Sedentary Lifestyle", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        prediabetes = st.selectbox("Prediabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        if st.form_submit_button("Predict"):
            try:
                inputs = {
                    "Age": age, "No of Pregnancy": no_pregnancy, "BMI": bmi, "HDL": hdl,
                    "Family History": family_history, "Large Child or Birth Default": large_child,
                    "PCOS": pcos, "Sys_BP": sys_bp, "Dia_BP": dia_bp, "OGTT": ogtt,
                    "Sedentary Lifestyle": sedentary, "Prediabetes": prediabetes,
                }
                result = predictor.predict_gdm(pd.DataFrame([inputs]))
                st.success(f"Prediction: {result}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab2:
    st.header("Maternal Health Risk Prediction")
    with st.form("maternal_form"):
        age = st.number_input("Age", min_value=10, max_value=70, value=25, key="m_age")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=120.0, key="m_sys")
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=80.0, key="m_dia")
        bs = st.number_input("Blood Sugar (mmol/L)", min_value=4.0, max_value=20.0, value=7.0)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70)

        if st.form_submit_button("Predict"):
            try:
                inputs = {
                    "Age": age, "SystolicBP": systolic_bp, "DiastolicBP": diastolic_bp,
                    "BS": bs, "HeartRate": heart_rate,
                }
                result = predictor.predict_maternal(pd.DataFrame([inputs]))
                st.success(f"Prediction: {result}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab3:
    st.header("Preeclampsia Risk Prediction")
    with st.form("preeclampsia_form"):
        gravida = st.number_input("Gravida", min_value=0, max_value=10, value=3)
        parity = st.number_input("Parity", min_value=0, max_value=10, value=1)
        gestational_age = st.number_input("Gestational Age (Weeks)", min_value=10.0, max_value=42.0, value=22.2)
        age = st.number_input("Age", min_value=18, max_value=50, value=25, key="p_age")
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=21.0, key="p_bmi")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="p_diab")
        hypertension = st.selectbox("Hypertension History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=110.0, key="p_sys")
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=70.0, key="p_dia")
        hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=9.3)
        fetal_weight = st.number_input("Fetal Weight (kg)", min_value=0.1, max_value=5.0, value=0.501)
        protein_uria = st.selectbox("Protein Uria", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        amniotic_fluid = st.number_input("Amniotic Fluid Levels (cm)", min_value=5.0, max_value=25.0, value=10.0)

        if st.form_submit_button("Predict"):
            try:
                inputs = {
                    "gravida": gravida, "parity": parity, "Gestational_Age_Weeks": gestational_age,
                    "Age": age, "BMI": bmi, "diabetes": diabetes, "Hypertension_History": hypertension,
                    "Systolic BP": systolic_bp, "Diastolic BP": diastolic_bp, "HB": hb,
                    "Fetal_Weight": fetal_weight, "Protein_Uria": protein_uria,
                    "Amniotic_Fluid_Levels": amniotic_fluid,
                }
                result = predictor.predict_preeclampsia(pd.DataFrame([inputs]))
                st.success(f"Prediction: {result}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
