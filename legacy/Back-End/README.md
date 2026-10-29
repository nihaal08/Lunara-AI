# Multi-Model Pregnancy Health Prediction System

## Overview
The **Multi-Model Pregnancy Health Prediction System** is a user-friendly web application built with **Streamlit** that integrates three machine learning models to assess pregnancy-related health risks. The system aims to aid healthcare professionals and pregnant individuals by providing early risk predictions for **Gestational Diabetes**, **Maternal Health Risks**, and **Preeclampsia**, facilitating timely interventions and better maternal-fetal outcomes.

---

## Aim
Develop an intuitive tool leveraging machine learning for early detection of:
- **Gestational Diabetes Mellitus (GDM)**
- **Maternal Health Risks** (low, mid, high)
- **Preeclampsia Risk** (low, mid, high)

This helps in proactive management and improves maternal and fetal health.

---

## Features

### 1. Gestational Diabetes Prediction
- **Outcome**: GDM or Non-GDM
- **Features**: Age, BMI, OGTT, Family History, HDL, Pregnancy history, PCOS, Blood Pressure, Lifestyle factors

### 2. Maternal Health Risk Prediction
- **Outcome**: Low, Mid, High risk
- **Features**: Age, Blood Pressure, Blood Sugar, Heart Rate

### 3. Preeclampsia Risk Prediction
- **Outcome**: Low, Mid, High risk
- **Features**: Gestational Age, BMI, Blood Pressure, Hemoglobin, Fetal Weight, Protein Uria, Amniotic Fluid Levels

### 4. Interactive Interface
- Built with **Streamlit**, featuring three dedicated tabs.
- User inputs via straightforward forms.
- Results are displayed with interpretive insights for actionable understanding.

---

## Directory Structure

```
C:.
├── Back-End
│   ├── Gestational_Diabetic
│   │       columns_gdm.pkl
│   │       gdm_model.pkl
│   │       Gestational_Diabetic.csv
│   │       label_encoder_gdm.pkl
│   │       model.py
│   │       scaler_gdm.pkl
│   ├── Maternal_Health
│   │       columns_maternal.pkl
│   │       label_encoder_maternal.pkl
│   │       Maternal Health Risk Data Set.csv
│   │       maternal_risk_model.pkl
│   │       model.py
│   │       scaler_maternal.pkl
│   ├── Preeclampsia
│   │       model.ipynb
│   │       model.py
│   │       Preeclampsia.csv
│   │       models
│   │           columns_improved.pkl
│   │           label_encoder_improved.pkl
│   │           preeclampsia_model_improved.pkl
│   │           scaler.pkl
│   └── streamlit
│       └── app.py
└── Documents
        Project-Synopsis.pdf
```

---

## Prerequisites

- **Python 3.8+** installed.
- Install required libraries:
  
  ```bash
  pip install streamlit pandas numpy scikit-learn xgboost imblearn joblib
  ```

- Ensure all model artifacts (`.pkl` files) are present in their respective directories:
  - `Back-End/Gestational_Diabetic/`
  - `Back-End/Maternal_Health/`
  - `Back-End/Preeclampsia/models/`

## Setup Instructions

1. **Clone or Download the Repository**
   
   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the Streamlit Directory**
   
   ```bash
   cd Back-End/streamlit
   ```

3. **Run the Application**
   
   ```bash
   streamlit run app.py
   ```

This will launch the app in your default browser at `http://localhost:8501`.

---

## Usage

### Launching the App
- Access the app via your browser.
- The interface contains three tabs:
  - **Gestational Diabetes Prediction**
  - **Maternal Health Risk Prediction**
  - **Preeclampsia Risk Prediction**

### Input Data
Fill out the form fields with patient-specific data:
- Example for GDM:
  - Age: 30
  - Number of Pregnancies: 2
  - BMI: 28.0
  - HDL: 50.0
  - Family History: No (0)
  - Large Child or Birth Defect: No (0)
  - PCOS: No (0)
  - Systolic BP: 120
  - Diastolic BP: 80
  - OGTT: 130
  - Sedentary Lifestyle: No (0)
  - Prediabetes: No (0)

### Viewing Predictions
- Click **Predict**.
- Results are shown with interpretive statements, e.g.,
  - *"Gestational Diabetes Prediction: Non GDM"*
  - *"This indicates no immediate concern for gestational diabetes..."*

---

## Model Details

| Model | Features | Engineering | Algorithm | Accuracy |
|---------|-----------------------------------------------------|------------------------------|------------|-----------|
| **Gestational Diabetes** | Age, Pregnancies, BMI, HDL, Family History, Large Child, PCOS, BP, OGTT, Lifestyle, Prediabetes | BP_ratio, Age_category, High_OGTT, BMI_category, OGTT_BMI_interaction, Age_OGTT_interaction | XGBoost (binary) | ~80% |
| **Maternal Health Risks** | Age, BP, Blood Sugar, Heart Rate | BP_ratio, Age_category, High_BS, SystolicBP_BS_interaction, Age_BS_interaction | XGBoost (multiclass) | ~80% |
| **Preeclampsia** | Gravida, Parity, Gestational Age, Age, BMI, Diabetes, Hypertension, BP, Hemoglobin, Fetal Weight, Protein Uria, Amniotic Fluid | BP_ratio, BMI_category | XGBoost (multiclass) | ~81% |

---

## Troubleshooting

- **Missing Model Files**:
  - Ensure all `.pkl` files are in their designated directories.
  - Re-train models if missing, using respective scripts:
    - GDM: `Back-End/Gestational_Diabetic/model.py`
    - Maternal Health: `Back-End/Maternal_Health/model.py`
    - Preeclampsia: `Back-End/Preeclampsia/model.ipynb` or `model.py`

- **Model Failures**:
  - Verify input ranges are realistic.
  - Check for proper dataset and label consistency.

- **Streamlit Not Launching**:
  - Confirm you're in the correct directory.
  - Install dependencies: `pip install -r requirements.txt` (create if necessary).

---

## Contributing
Contributions are welcome! To contribute:
- Fork the repository.
- Create a feature or bug fix branch:
  ```bash
  git checkout -b feature-name
  ```
- Make your changes, test thoroughly.
- Submit a pull request with detailed descriptions.

---

## License
[Add your chosen license here, e.g., MIT License, Apache License 2.0]

---

## Acknowledgments
- **Datasets**:
  - Gestational Diabetes: `Gestational_Diabetic.csv`
  - Maternal Health: `Maternal Health Risk Data Set.csv`
  - Preeclampsia: `Preeclampsia.csv`

- **Tools & Libraries**:
  - Streamlit for UI.
  - XGBoost for modeling.
  - scikit-learn, imblearn for preprocessing.

---

*This README aims to guide users through setup, usage, and understanding of the Multi-Model Pregnancy Health Prediction System.*