"""Unified ML inference service for all maternal health models."""

import logging
import sys
import types
from pathlib import Path

import joblib
import numpy
import pandas as pd

# Compatibility shim: allow loading artifacts serialized across different NumPy versions (e.g. NumPy 2.x <-> 1.x)
if hasattr(numpy, "core") and "numpy._core" not in sys.modules:
    core_shim = types.ModuleType("numpy._core")
    core_shim.__dict__.update(numpy.core.__dict__)
    sys.modules["numpy._core"] = core_shim
    if hasattr(numpy.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    if hasattr(numpy.core, "_multiarray_umath"):
        sys.modules["numpy._core._multiarray_umath"] = numpy.core._multiarray_umath

from ml.gestational_diabetes.features import GDM_NUMERIC_COLS, engineer_gdm_features
from ml.maternal_health.features import engineer_maternal_features
from ml.paths import (
    GDM_ARTIFACTS,
    GESTATIONAL_DIABETES_DIR,
    MATERNAL_ARTIFACTS,
    MATERNAL_HEALTH_DIR,
    PREECLAMPSIA_ARTIFACTS,
    PREECLAMPSIA_DIR,
)
from ml.preeclampsia.features import engineer_preeclampsia_features

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Loads serialized models and runs inference with consistent preprocessing."""

    def __init__(self, models_root: str | Path | None = None):
        self.models_root = Path(models_root) if models_root else Path(__file__).resolve().parent
        self._artifact_cache: dict[str, dict] = {}

    def _artifact_dir(self, model_key: str) -> Path:
        mapping = {
            "gdm": GESTATIONAL_DIABETES_DIR / "artifacts",
            "maternal": MATERNAL_HEALTH_DIR / "artifacts",
            "preeclampsia": PREECLAMPSIA_DIR / "artifacts",
        }
        return mapping[model_key]

    def _load_artifacts(self, model_key: str, filenames: dict[str, str]) -> dict:
        if model_key in self._artifact_cache:
            return self._artifact_cache[model_key]

        artifact_dir = self._artifact_dir(model_key)
        loaded = {}
        for name, filename in filenames.items():
            path = artifact_dir / filename
            if not path.exists():
                # Support legacy artifact locations
                legacy_path = self._resolve_legacy_path(model_key, name)
                if legacy_path and legacy_path.exists():
                    path = legacy_path
                else:
                    raise FileNotFoundError(
                        f"Model artifact not found: {path}. "
                        f"Train models with: python scripts/train_all_models.py"
                    )
            loaded[name] = joblib.load(path)

        self._artifact_cache[model_key] = loaded
        logger.info("Loaded %s model artifacts from %s", model_key, artifact_dir)
        return loaded

    @staticmethod
    def _resolve_legacy_path(model_key: str, artifact_name: str) -> Path | None:
        """Map legacy artifact locations for backward compatibility."""
        legacy_map = {
            ("preeclampsia", "model"): [
                Path("legacy/Back-End/Preeclampsia/models/preeclampsia_model_improved.pkl"),
                Path("Back-End/Preeclampsia/models/preeclampsia_model_improved.pkl"),
            ],
            ("preeclampsia", "label_encoder"): [
                Path("legacy/Back-End/Preeclampsia/models/label_encoder_improved.pkl"),
                Path("Back-End/Preeclampsia/models/label_encoder_improved.pkl"),
            ],
            ("preeclampsia", "scaler"): [
                Path("legacy/Back-End/Preeclampsia/models/scaler.pkl"),
                Path("Back-End/Preeclampsia/models/scaler.pkl"),
            ],
            ("preeclampsia", "columns"): [
                Path("legacy/Back-End/Preeclampsia/models/columns_improved.pkl"),
                Path("Back-End/Preeclampsia/models/columns_improved.pkl"),
            ],
            ("gdm", "model"): [
                Path("legacy/Back-End/Gestational_Diabetic/gdm_model.pkl"),
                Path("Back-End/Gestational_Diabetic/gdm_model.pkl"),
            ],
            ("gdm", "label_encoder"): [
                Path("legacy/Back-End/Gestational_Diabetic/label_encoder_gdm.pkl"),
                Path("Back-End/Gestational_Diabetic/label_encoder_gdm.pkl"),
            ],
            ("gdm", "scaler"): [
                Path("legacy/Back-End/Gestational_Diabetic/scaler_gdm.pkl"),
                Path("Back-End/Gestational_Diabetic/scaler_gdm.pkl"),
            ],
            ("gdm", "columns"): [
                Path("legacy/Back-End/Gestational_Diabetic/columns_gdm.pkl"),
                Path("Back-End/Gestational_Diabetic/columns_gdm.pkl"),
            ],
            ("maternal", "model"): [
                Path("legacy/Back-End/Maternal_Health/maternal_risk_model.pkl"),
                Path("Back-End/Maternal_Health/maternal_risk_model.pkl"),
            ],
            ("maternal", "label_encoder"): [
                Path("legacy/Back-End/Maternal_Health/label_encoder_maternal.pkl"),
                Path("Back-End/Maternal_Health/label_encoder_maternal.pkl"),
            ],
            ("maternal", "scaler"): [
                Path("legacy/Back-End/Maternal_Health/scaler_maternal.pkl"),
                Path("Back-End/Maternal_Health/scaler_maternal.pkl"),
            ],
            ("maternal", "columns"): [
                Path("legacy/Back-End/Maternal_Health/columns_maternal.pkl"),
                Path("Back-End/Maternal_Health/columns_maternal.pkl"),
            ],
        }
        candidates = legacy_map.get((model_key, artifact_name), [])
        project_root = Path(__file__).resolve().parent.parent
        for rel in candidates:
            cand_path = project_root / rel
            if cand_path.exists():
                return cand_path
        return None

    def predict_gdm(self, new_data: pd.DataFrame) -> str:
        """Predict Gestational Diabetes Mellitus risk."""
        artifacts = self._load_artifacts("gdm", GDM_ARTIFACTS)
        df = engineer_gdm_features(new_data)
        df = df.reindex(columns=artifacts["columns"], fill_value=0)
        df[GDM_NUMERIC_COLS] = artifacts["scaler"].transform(df[GDM_NUMERIC_COLS])
        predictions = artifacts["model"].predict(df)
        return artifacts["label_encoder"].inverse_transform(predictions)[0]

    def predict_maternal(self, new_data: pd.DataFrame) -> str:
        """Predict Maternal Health Risk level."""
        artifacts = self._load_artifacts("maternal", MATERNAL_ARTIFACTS)
        df = engineer_maternal_features(new_data)
        df = df.reindex(columns=artifacts["columns"], fill_value=0)
        scaled = artifacts["scaler"].transform(df)
        predictions = artifacts["model"].predict(scaled)
        return artifacts["label_encoder"].inverse_transform(predictions)[0]

    def predict_preeclampsia(self, new_data: pd.DataFrame) -> str:
        """Predict Preeclampsia risk level (low / mid / high)."""
        artifacts = self._load_artifacts("preeclampsia", PREECLAMPSIA_ARTIFACTS)
        df = engineer_preeclampsia_features(new_data)
        df = df.reindex(columns=artifacts["columns"], fill_value=0)
        scaled = artifacts["scaler"].transform(df)
        predictions = artifacts["model"].predict(scaled)
        return artifacts["label_encoder"].inverse_transform(predictions)[0]
