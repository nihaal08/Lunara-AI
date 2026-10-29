"""
Train the Gestational Diabetes Mellitus (GDM) XGBoost classifier.

Usage:
    python -m ml.gestational_diabetes.train
"""

import warnings
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.gestational_diabetes.features import engineer_gdm_features
from ml.paths import GDM_ARTIFACTS, GESTATIONAL_DIABETES_DIR

warnings.filterwarnings("ignore")

DATA_PATH = GESTATIONAL_DIABETES_DIR / "data" / "Gestational_Diabetic.csv"
ARTIFACTS_DIR = GESTATIONAL_DIABETES_DIR / "artifacts"


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and apply training-time preprocessing."""
    df = pd.read_csv(DATA_PATH)

    columns_to_drop = [
        "Case Number",
        "Gestation in previous Pregnancy",
        "unexplained prenetal loss",
        "Hemoglobin",
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df = df.rename(
        columns={
            "Sys BP": "Sys_BP",
            "Dia BP": "Dia_BP",
            "Class Label(GDM /Non GDM)": "Class_Label",
        }
    )

    df["Class_Label"] = df["Class_Label"].astype(str).str.strip()
    df["Class_Label"] = df["Class_Label"].replace({"0": "GDM", "1": "Non GDM"})
    df = df[df["Class_Label"].isin(["GDM", "Non GDM"])].copy()

    numeric_columns = ["Age", "No of Pregnancy", "BMI", "HDL", "Sys_BP", "Dia_BP", "OGTT"]
    categorical_columns = [
        "Family History",
        "Large Child or Birth Default",
        "PCOS",
        "Sedentary Lifestyle",
        "Prediabetes",
    ]

    num_imputer = SimpleImputer(strategy="median")
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.25 * iqr, upper=q3 + 1.25 * iqr)
    df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    for col in categorical_columns:
        df[col] = df[col].astype(int)

    df = engineer_gdm_features(df)

    if df.isna().any().any():
        df = df.fillna(df.median(numeric_only=True))

    label_encoder = LabelEncoder()
    df["Class_Label"] = label_encoder.fit_transform(df["Class_Label"])

    x = df.drop("Class_Label", axis=1)
    y = df["Class_Label"]

    scaler = StandardScaler()
    scale_cols = numeric_columns + ["BP_ratio", "OGTT_BMI_interaction", "Age_OGTT_interaction"]
    x[scale_cols] = scaler.fit_transform(x[scale_cols])

    return x, y, label_encoder, scaler


def train() -> None:
    """Train GDM model and persist artifacts."""
    x, y, label_encoder, scaler = load_and_preprocess()

    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(
        x_resampled, y_resampled, test_size=0.2, random_state=42
    )

    model = XGBClassifier(random_state=42, eval_metric="logloss")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring="f1", n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, x_train, y_train, cv=10, scoring="accuracy")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    y_pred = best_model.predict(x_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS_DIR / GDM_ARTIFACTS["model"])
    joblib.dump(label_encoder, ARTIFACTS_DIR / GDM_ARTIFACTS["label_encoder"])
    joblib.dump(scaler, ARTIFACTS_DIR / GDM_ARTIFACTS["scaler"])
    joblib.dump(x_train.columns, ARTIFACTS_DIR / GDM_ARTIFACTS["columns"])
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train()
