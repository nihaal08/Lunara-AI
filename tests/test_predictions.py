"""Tests for ML inference pipeline."""

import unittest
import pandas as pd
from ml.predictor import ModelPredictor


class MLPredictorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = ModelPredictor()

    def test_gdm_prediction(self):
        data = pd.DataFrame([{
            "Age": 28, "No of Pregnancy": 1, "BMI": 22.0, "HDL": 55,
            "Family History": 0, "Large Child or Birth Default": 0,
            "PCOS": 0, "Sys_BP": 115, "Dia_BP": 75, "OGTT": 95,
            "Sedentary Lifestyle": 0, "Prediabetes": 0
        }])
        result = self.predictor.predict_gdm(data)
        self.assertIn(result, ["GDM", "Non GDM"])

    def test_maternal_prediction(self):
        data = pd.DataFrame([{
            "Age": 24, "SystolicBP": 110, "DiastolicBP": 70, "BS": 6.8, "HeartRate": 72
        }])
        result = self.predictor.predict_maternal(data)
        self.assertIn(result.lower(), ["low risk", "mid risk", "high risk"])

    def test_preeclampsia_prediction(self):
        data = pd.DataFrame([{
            "gravida": 2, "parity": 1, "Gestational_Age_Weeks": 20.0, "Age": 26, "BMI": 22.5,
            "diabetes": 0, "Hypertension_History": 0, "Systolic BP": 110, "Diastolic BP": 70,
            "HB": 11.0, "Fetal_Weight": 0.450, "Protein_Uria": 0, "Amniotic_Fluid_Levels": 12.0
        }])
        result = self.predictor.predict_preeclampsia(data)
        self.assertIn(result.lower(), ["low", "mid", "high"])


if __name__ == "__main__":
    unittest.main()
