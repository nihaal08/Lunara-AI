"""Feature engineering for Maternal Health Risk inference."""

import pandas as pd


def engineer_maternal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during maternal health model training."""
    df = data.copy()

    df["BP_ratio"] = df["SystolicBP"] / df["DiastolicBP"].replace(0, 1)
    df["Age_category"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 100],
        labels=[0, 1, 2],
        include_lowest=True,
    ).cat.codes
    df["Age_category"] = df["Age_category"].replace(-1, 1)
    df["High_BS"] = (df["BS"] > 11).astype(int)
    df["SystolicBP_BS_interaction"] = df["SystolicBP"] * df["BS"]
    df["Age_BS_interaction"] = df["Age"] * df["BS"]

    return df
