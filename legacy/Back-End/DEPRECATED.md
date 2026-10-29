# Deprecated — Use Root Project Instead

This folder contains the **legacy** ML training scripts and Streamlit demo.

The project has been refactored to a production-ready layout at the repository root:

- **Train models:** `python scripts/train_all_models.py`
- **New ML code:** `ml/`
- **Streamlit demo:** `streamlit_app/app.py`
- **Documentation:** `README.md`

Existing `.pkl` artifacts in this folder are still supported via legacy path fallback in `ml/predictor.py`. Copy them to `ml/*/artifacts/` for the new convention.
