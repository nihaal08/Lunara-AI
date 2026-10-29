"""Feature engineering for Preeclampsia risk inference."""

import numpy as np
import pandas as pd


def engineer_preeclampsia_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during preeclampsia model training."""
    df = data.copy()

    df["BP_ratio"] = df["Systolic BP"] / df["Diastolic BP"].replace(0, 1)
    df["BP_ratio"] = df["BP_ratio"].replace(
        [np.inf, -np.inf], df["BP_ratio"].median()
    )
    df["BMI_category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).cat.codes
    df["BMI_category"] = df["BMI_category"].replace(-1, 1)

    return df
