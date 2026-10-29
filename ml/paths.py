"""Centralized paths for ML datasets and serialized model artifacts."""

import os
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ML_ROOT.parent

GESTATIONAL_DIABETES_DIR = ML_ROOT / "gestational_diabetes"
MATERNAL_HEALTH_DIR = ML_ROOT / "maternal_health"
PREECLAMPSIA_DIR = ML_ROOT / "preeclampsia"

# Artifact filenames (consistent across training and inference)
GDM_ARTIFACTS = {
    "model": "gdm_model.pkl",
    "label_encoder": "label_encoder_gdm.pkl",
    "scaler": "scaler_gdm.pkl",
    "columns": "columns_gdm.pkl",
}

MATERNAL_ARTIFACTS = {
    "model": "maternal_risk_model.pkl",
    "label_encoder": "label_encoder_maternal.pkl",
    "scaler": "scaler_maternal.pkl",
    "columns": "columns_maternal.pkl",
}

PREECLAMPSIA_ARTIFACTS = {
    "model": "preeclampsia_model.pkl",
    "label_encoder": "label_encoder_preeclampsia.pkl",
    "scaler": "scaler_preeclampsia.pkl",
    "columns": "columns_preeclampsia.pkl",
}


def get_artifacts_dir(model_name: str) -> Path:
    """Return the artifacts directory for a given model."""
    mapping = {
        "gestational_diabetes": GESTATIONAL_DIABETES_DIR / "artifacts",
        "maternal_health": MATERNAL_HEALTH_DIR / "artifacts",
        "preeclampsia": PREECLAMPSIA_DIR / "artifacts",
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model: {model_name}")
    return mapping[model_name]


def get_models_path() -> Path:
    """Return the root ML directory used by the Flask app for inference."""
    return Path(os.environ.get("MODELS_PATH", ML_ROOT))
