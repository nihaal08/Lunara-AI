"""Feature engineering for Gestational Diabetes Mellitus (GDM) inference."""

import pandas as pd


def engineer_gdm_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during GDM model training."""
    df = data.copy()

    df["BP_ratio"] = df["Sys_BP"] / df["Dia_BP"].replace(0, 1)
    df["Age_category"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 100],
        labels=[0, 1, 2],
        include_lowest=True,
    ).cat.codes
    df["Age_category"] = df["Age_category"].replace(-1, 1)
    df["High_OGTT"] = (df["OGTT"] > 140).astype(int)
    df["BMI_category"] = pd.cut(
        df["BMI"],
        bins=[0, 25, 30, 100],
        labels=[0, 1, 2],
        include_lowest=True,
    ).cat.codes
    df["BMI_category"] = df["BMI_category"].replace(-1, 1)
    df["OGTT_BMI_interaction"] = df["OGTT"] * df["BMI"]
    df["Age_OGTT_interaction"] = df["Age"] * df["OGTT"]

    return df


GDM_NUMERIC_COLS = [
    "Age",
    "No of Pregnancy",
    "BMI",
    "HDL",
    "Sys_BP",
    "Dia_BP",
    "OGTT",
    "BP_ratio",
    "OGTT_BMI_interaction",
    "Age_OGTT_interaction",
]
