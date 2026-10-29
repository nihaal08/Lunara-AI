# Legacy Codebase Archive

This directory preserves the original, monolithic codebase and exploratory development scripts of the LUNARA-AI platform. **No files have been removed or deleted.**

## Directory Structure

```
legacy/
├── Back-End/
│   ├── Gestational_Diabetic/     # Original GDM model training scripts & artifacts
│   ├── Maternal_Health/          # Original Maternal Risk scripts & artifacts
│   ├── Preeclampsia/             # Original Preeclampsia notebook & models
│   ├── streamlit/                # Original monolithic Streamlit demo
│   └── README.md
│
└── Front-End/
    ├── app.py                    # Original monolithic Flask web application
    ├── config.py                 # Original configuration file
    ├── database.db               # Original SQLite database instance
    ├── utils.py                  # Original utility helpers & PDF generation
    ├── models/                   # Original model copies
    ├── static/                   # Static CSS and JS assets
    └── templates/                # Original Jinja2 HTML templates
```

## Active Production Codebase

For current development and production usage, use the root-aligned modular project structure:
- **Web Application:** `app/` (Modular Flask application with Blueprints & Service layer)
- **Machine Learning:** `ml/` (Centralized pipelines, feature engineering, and `artifacts/`)
- **Research Notebooks:** `notebooks/` (Exploratory data analysis & model experiments)
- **Model Training:** `python scripts/train_all_models.py`
- **Application Runner:** `python run.py`
- **Streamlit Demo:** `streamlit run streamlit_app/app.py`
