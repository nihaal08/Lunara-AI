import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Function to load Gestational Diabetes artifacts
def load_gdm_artifacts():
    try:
        model = joblib.load('../Gestational_Diabetic/gdm_model.pkl')
        scaler = joblib.load('../Gestational_Diabetic/scaler_gdm.pkl')
        label_encoder = joblib.load('../Gestational_Diabetic/label_encoder_gdm.pkl')
        columns = joblib.load('../Gestational_Diabetic/columns_gdm.pkl')
        return model, scaler, label_encoder, columns
    except FileNotFoundError:
        st.error("Gestational Diabetes model artifacts not found. Ensure all .pkl files are in the 'Gestational_Diabetic' directory.")
        return None, None, None, None

# Function to load Maternal Health artifacts
def load_maternal_artifacts():
    try:
        model = joblib.load('../Maternal_Health/maternal_risk_model.pkl')
        scaler = joblib.load('../Maternal_Health/scaler_maternal.pkl')
        label_encoder = joblib.load('../Maternal_Health/label_encoder_maternal.pkl')
        columns = joblib.load('../Maternal_Health/columns_maternal.pkl')
        return model, scaler, label_encoder, columns
    except FileNotFoundError:
        st.error("Maternal Health model artifacts not found. Ensure all .pkl files are in the 'Maternal_Health' directory.")
        return None, None, None, None

# Function to load Preeclampsia artifacts
def load_preeclampsia_artifacts():
    try:
        model = joblib.load('../Preeclampsia/models/preeclampsia_model_improved.pkl')
        scaler = joblib.load('../Preeclampsia/models/scaler.pkl')
        label_encoder = joblib.load('../Preeclampsia/models/label_encoder_improved.pkl')
        columns = joblib.load('../Preeclampsia/models/columns_improved.pkl')
        return model, scaler, label_encoder, columns
    except FileNotFoundError:
        st.error("Preeclampsia model artifacts not found. Ensure all .pkl files are in the 'Preeclampsia/models' directory.")
        return None, None, None, None

# Function to preprocess and predict for Gestational Diabetes
def predict_gdm(model, scaler, label_encoder, columns, input_data):
    df = pd.DataFrame([input_data])
    # Feature engineering
    df['BP_ratio'] = df['Sys_BP'] / df['Dia_BP'].replace(0, 1)
    df['Age_category'] = pd.cut(df['Age'], bins=[0, 25, 35, 100], labels=[0, 1, 2], include_lowest=True)
    df['Age_category'] = df['Age_category'].cat.codes
    df['Age_category'] = df['Age_category'].replace(-1, 1)
    df['High_OGTT'] = (df['OGTT'] > 140).astype(int)
    df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 25, 30, 100], labels=[0, 1, 2], include_lowest=True)
    df['BMI_category'] = df['BMI_category'].cat.codes
    df['BMI_category'] = df['BMI_category'].replace(-1, 1)
    df['OGTT_BMI_interaction'] = df['OGTT'] * df['BMI']
    df['Age_OGTT_interaction'] = df['Age'] * df['OGTT']
    df = df.reindex(columns=columns, fill_value=0)
    numeric_cols = ['Age', 'No of Pregnancy', 'BMI', 'HDL', 'Sys_BP', 'Dia_BP', 'OGTT', 
                    'BP_ratio', 'OGTT_BMI_interaction', 'Age_OGTT_interaction']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    pred = model.predict(df)[0]
    return label_encoder.inverse_transform([pred])[0]

# Function to preprocess and predict for Maternal Health
def predict_maternal(model, scaler, label_encoder, columns, input_data):
    df = pd.DataFrame([input_data])
    # Feature engineering
    df['BP_ratio'] = df['SystolicBP'] / df['DiastolicBP'].replace(0, 1)
    df['Age_category'] = pd.cut(df['Age'], bins=[0, 18, 35, 100], labels=[0, 1, 2], include_lowest=True)
    df['Age_category'] = df['Age_category'].cat.codes
    df['Age_category'] = df['Age_category'].replace(-1, 1)
    df['High_BS'] = (df['BS'] > 11).astype(int)
    df['SystolicBP_BS_interaction'] = df['SystolicBP'] * df['BS']
    df['Age_BS_interaction'] = df['Age'] * df['BS']
    df = df.reindex(columns=columns, fill_value=0)
    df = scaler.transform(df)
    pred = model.predict(df)[0]
    return label_encoder.inverse_transform([pred])[0]

# Function to preprocess and predict for Preeclampsia
def predict_preeclampsia(model, scaler, label_encoder, columns, input_data):
    df = pd.DataFrame([input_data])
    # Feature engineering
    df['BP_ratio'] = df['Systolic BP'] / df['Diastolic BP'].replace(0, 1)
    df['BP_ratio'] = df['BP_ratio'].replace([np.inf, -np.inf], df['BP_ratio'].median())
    df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3], include_lowest=True)
    df['BMI_category'] = df['BMI_category'].cat.codes
    df['BMI_category'] = df['BMI_category'].replace(-1, 1)
    df = df.reindex(columns=columns, fill_value=0)
    df = scaler.transform(df)
    pred = model.predict(df)[0]
    return label_encoder.inverse_transform([pred])[0]

# Streamlit App
st.title("Multi-Model Pregnancy Health Prediction System")
st.markdown("Enter patient details to predict risks for Gestational Diabetes, Maternal Health, or Preeclampsia.")

# Tabs for model selection
tab1, tab2, tab3 = st.tabs(["Gestational Diabetes Prediction", "Maternal Health Risk Prediction", "Preeclampsia Risk Prediction"])

# Gestational Diabetes Prediction Tab
with tab1:
    st.header("Gestational Diabetes Prediction")
    gdm_model, gdm_scaler, gdm_label_encoder, gdm_columns = load_gdm_artifacts()
    if gdm_model:
        with st.form("gdm_form"):
            age = st.number_input("Age", min_value=18, max_value=50, value=30)
            no_pregnancy = st.number_input("Number of Pregnancies", min_value=0, max_value=10, value=2)
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=28.0)
            hdl = st.number_input("HDL (mg/dL)", min_value=20.0, max_value=100.0, value=50.0)
            family_history = st.selectbox("Family History of Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            large_child = st.selectbox("Large Child or Birth Defect", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            pcos = st.selectbox("PCOS", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            sys_bp = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=200.0, value=120.0)
            dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=80.0)
            ogtt = st.number_input("OGTT (mg/dL)", min_value=50.0, max_value=300.0, value=130.0)
            sedentary = st.selectbox("Sedentary Lifestyle", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            prediabetes = st.selectbox("Prediabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            submit = st.form_submit_button("Predict")
            if submit:
                input_data = {
                    'Age': age, 'No of Pregnancy': no_pregnancy, 'BMI': bmi, 'HDL': hdl,
                    'Family History': family_history, 'Large Child or Birth Default': large_child,
                    'PCOS': pcos, 'Sys_BP': sys_bp, 'Dia_BP': dia_bp, 'OGTT': ogtt,
                    'Sedentary Lifestyle': sedentary, 'Prediabetes': prediabetes
                }
                pred = predict_gdm(gdm_model, gdm_scaler, gdm_label_encoder, gdm_columns, input_data)
                st.success(f"Gestational Diabetes Prediction: {pred}")
                st.markdown(
                    f"**Interpretation**: This indicates {'a risk for gestational diabetes. Consult an obstetrician for further evaluation.' if pred == 'GDM' else 'no immediate concern for gestational diabetes, but continue monitoring.'}"
                )

# Maternal Health Risk Prediction Tab
with tab2:
    st.header("Maternal Health Risk Prediction")
    maternal_model, maternal_scaler, maternal_label_encoder, maternal_columns = load_maternal_artifacts()
    if maternal_model:
        with st.form("maternal_form"):
            age = st.number_input("Age", min_value=10, max_value=70, value=25)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=120.0)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=80.0)
            bs = st.number_input("Blood Sugar (mmol/L)", min_value=4.0, max_value=20.0, value=7.0)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70)
            submit = st.form_submit_button("Predict")
            if submit:
                input_data = {
                    'Age': age, 'SystolicBP': systolic_bp, 'DiastolicBP': diastolic_bp,
                    'BS': bs, 'HeartRate': heart_rate
                }
                pred = predict_maternal(maternal_model, maternal_scaler, maternal_label_encoder, maternal_columns, input_data)
                st.success(f"Maternal Health Risk: {pred}")
                st.markdown(
                    f"**Interpretation**: This indicates {'a high-risk pregnancy requiring immediate medical attention.' if pred == 'high risk' else 'a moderate risk level; consult a healthcare provider.' if pred == 'mid risk' else 'a low risk level, but regular monitoring is recommended.'}"
                )

# Preeclampsia Risk Prediction Tab
with tab3:
    st.header("Preeclampsia Risk Prediction")
    preeclampsia_model, preeclampsia_scaler, preeclampsia_label_encoder, preeclampsia_columns = load_preeclampsia_artifacts()
    if preeclampsia_model:
        with st.form("preeclampsia_form"):
            gravida = st.number_input("Gravida (Number of Pregnancies)", min_value=0, max_value=10, value=3)
            parity = st.number_input("Parity (Number of Births)", min_value=0, max_value=10, value=1)
            gestational_age = st.number_input("Gestational Age (Weeks)", min_value=10.0, max_value=42.0, value=22.2)
            age = st.number_input("Age", min_value=18, max_value=50, value=25)
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=21.0)
            diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hypertension = st.selectbox("Hypertension History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=110.0)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=70.0)
            hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=9.3)
            fetal_weight = st.number_input("Fetal Weight (kg)", min_value=0.1, max_value=5.0, value=0.501)
            protein_uria = st.selectbox("Protein Uria", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            amniotic_fluid = st.number_input("Amniotic Fluid Levels (cm)", min_value=5.0, max_value=25.0, value=10.0)
            submit = st.form_submit_button("Predict")
            if submit:
                input_data = {
                    'gravida': gravida, 'parity': parity, 'Gestational_Age_Weeks': gestational_age,
                    'Age': age, 'BMI': bmi, 'diabetes': diabetes, 'Hypertension_History': hypertension,
                    'Systolic BP': systolic_bp, 'Diastolic BP': diastolic_bp, 'HB': hb,
                    'Fetal_Weight': fetal_weight, 'Protein_Uria': protein_uria,
                    'Amniotic_Fluid_Levels': amniotic_fluid
                }
                pred = predict_preeclampsia(preeclampsia_model, preeclampsia_scaler, preeclampsia_label_encoder, preeclampsia_columns, input_data)
                st.success(f"Preeclampsia Risk Prediction: {pred}")
                st.markdown(
                    f"**Interpretation**: This indicates {'a high risk of preeclampsia. Immediate medical consultation is advised.' if pred == 'high' else 'a moderate risk level; consult a healthcare provider.' if pred == 'mid' else 'a low risk level, but regular monitoring is recommended.'}"
                )