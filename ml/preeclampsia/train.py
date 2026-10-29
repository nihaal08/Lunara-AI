"""
Train the Preeclampsia Risk XGBoost classifier.

Usage:
    python -m ml.preeclampsia.train
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.paths import PREECLAMPSIA_ARTIFACTS, PREECLAMPSIA_DIR
from ml.preeclampsia.features import engineer_preeclampsia_features

warnings.filterwarnings("ignore")

DATA_PATH = PREECLAMPSIA_DIR / "data" / "Preeclampsia.csv"
ARTIFACTS_DIR = PREECLAMPSIA_DIR / "artifacts"


def load_and_preprocess():
    """Load dataset and apply training-time preprocessing."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 12"], errors="ignore")
    df.columns = df.columns.str.strip()
    df.rename(
        columns={
            "Age (yrs)": "Age",
            "BMI  [kg/m²]": "BMI",
            "History of hypertension (y/n)": "Hypertension_History",
            "gestational age (weeks)": "Gestational_Age_Weeks",
            "fetal weight(kgs)": "Fetal_Weight",
            "Protien Uria": "Protein_Uria",
            "Uterine Artery Doppler Resistance Index (RI)": "Uterine_RI",
            "Uterine Artery Doppler Pulsatility Index (PI": "Uterine_PI",
            "amniotic fluid levels(cm)": "Amniotic_Fluid_Levels",
        },
        inplace=True,
    )

    numeric_cols = [
        "gravida", "parity", "Gestational_Age_Weeks", "Age", "BMI",
        "Systolic BP", "Diastolic BP", "HB", "Fetal_Weight",
        "Uterine_RI", "Uterine_PI", "Amniotic_Fluid_Levels",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Age", "Systolic BP", "Diastolic BP", "Fetal_Weight"], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[df["gravida"] != "gravida"]
    df = df.drop(columns=["Uterine_RI", "Uterine_PI"], errors="ignore")
    df.columns = df.columns.str.strip()

    numeric_columns = [
        "gravida", "parity", "Gestational_Age_Weeks", "Age", "BMI",
        "Systolic BP", "Diastolic BP", "HB", "Fetal_Weight", "Amniotic_Fluid_Levels",
    ]
    num_imputer = SimpleImputer(strategy="median")
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

    binary_columns = ["diabetes", "Hypertension_History", "Protein_Uria"]
    bin_imputer = SimpleImputer(strategy="most_frequent")
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: 1 if str(x).lower() in ["y", "1", "yes", "true"]
                else 0 if str(x).lower() in ["n", "0", "no", "false"]
                else np.nan
            )
    df[binary_columns] = bin_imputer.fit_transform(df[binary_columns])

    df = engineer_preeclampsia_features(df)

    if df.isna().any().any():
        df = df.fillna(df.median(numeric_only=True))

    label_encoder = LabelEncoder()
    df["Risk_level"] = label_encoder.fit_transform(df["Risk_level"])

    x = df.drop("Risk_level", axis=1)
    y = df["Risk_level"]

    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    return x, y, label_encoder, scaler


def train() -> None:
    """Train preeclampsia model and persist artifacts."""
    x, y, label_encoder, scaler = load_and_preprocess()

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(
        x_resampled, y_resampled, test_size=0.2, random_state=42
    )

    model = XGBClassifier(random_state=42, eval_metric="mlogloss")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring="accuracy")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    y_pred = best_model.predict(x_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_DIR / PREECLAMPSIA_ARTIFACTS["model"])
    joblib.dump(label_encoder, ARTIFACTS_DIR / PREECLAMPSIA_ARTIFACTS["label_encoder"])
    joblib.dump(scaler, ARTIFACTS_DIR / PREECLAMPSIA_ARTIFACTS["scaler"])
    joblib.dump(x_train.columns, ARTIFACTS_DIR / PREECLAMPSIA_ARTIFACTS["columns"])
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train()
