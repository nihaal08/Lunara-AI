# LUNARA-AI: Maternal Healthcare Prediction Platform

LUNARA-AI is a full-stack machine learning application for early risk prediction of maternal health conditions:

- **Gestational Diabetes Mellitus (GDM)**
- **Maternal Health Risk** (low / mid / high)
- **Preeclampsia Risk** (low / mid / high)

The platform uses XGBoost classifiers with SMOTE-based imbalance handling, a Flask web application with user authentication, prediction history, PDF reports, and an optional Streamlit demo interface.

---

## Project Structure

```
LUNARA-AI/
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
│
├── app/                        # Flask web application (backend + frontend)
│   ├── __init__.py             # App factory
│   ├── config.py               # Environment-based configuration
│   ├── database.py             # SQLite schema & connection manager
│   ├── decorators.py           # Auth decorators
│   ├── logging_config.py       # Rotating file + console logging
│   ├── routes/                 # Blueprint route modules
│   │   ├── main.py             # Home, about, contact
│   │   ├── auth.py             # Login, signup, logout
│   │   ├── predictions.py      # ML prediction forms
│   │   ├── history.py          # History & PDF export
│   │   ├── admin.py            # Admin dashboard
│   │   └── api.py              # Theme API
│   ├── services/               # Business logic layer
│   │   ├── auth_service.py
│   │   ├── prediction_service.py
│   │   ├── pdf_service.py
│   │   └── audit_service.py
│   ├── static/                 # CSS & JavaScript
│   └── templates/              # Jinja2 HTML templates
│
├── ml/                         # Machine learning layer
│   ├── predictor.py            # Unified inference service
│   ├── paths.py                # Artifact path configuration
│   ├── gestational_diabetes/
│   │   ├── train.py            # Training pipeline
│   │   ├── features.py         # Feature engineering
│   │   ├── data/               # Training CSV
│   │   └── artifacts/          # Serialized models (.pkl)
│   ├── maternal_health/
│   │   ├── train.py
│   │   ├── features.py
│   │   ├── data/
│   │   └── artifacts/
│   └── preeclampsia/
│       ├── train.py
│       ├── features.py
│       ├── data/
│       └── artifacts/
│
├── notebooks/                  # Exploratory research notebooks
│   └── preeclampsia_model.ipynb
│
├── tests/                      # Automated unit tests
│   ├── test_app.py
│   └── test_predictions.py
│
├── scripts/
│   └── train_all_models.py     # Train all three models
│
├── streamlit_app/
│   └── app.py                  # Optional Streamlit demo UI
│
├── legacy/                     # Preserved original archive (No files deleted)
│   ├── Back-End/               # Original training scripts and legacy artifacts
│   └── Front-End/              # Original monolithic Flask application
│
└── instance/                   # Runtime data (gitignored)
    ├── database.db             # SQLite database
    └── lunara.log              # Application logs
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone <repository-url>
cd LUNARA-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and set a strong `SECRET_KEY` before deploying to production.

---

## Environment Setup

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | `development`, `production`, or `testing` |
| `SECRET_KEY` | (dev default) | Flask session signing key |
| `DATABASE_PATH` | `instance/database.db` | SQLite database file path |
| `MODELS_PATH` | `ml` | Root directory for ML artifacts |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `FLASK_HOST` | `0.0.0.0` | Server bind address |
| `FLASK_PORT` | `5000` | Server port |

---

## Database Setup

The database is created automatically on first run. Tables:

- **users** — accounts with hashed passwords and admin flag
- **predictions** — stored prediction inputs and results per user
- **audit_log** — login, logout, and admin action records

To reset the database, delete `instance/database.db` and restart the app.

### Create an admin user

Register a normal account, then promote it in SQLite:

```bash
sqlite3 instance/database.db "UPDATE users SET is_admin = 1 WHERE username = 'your_username';"
```

---

## Model Training

Train all models (requires datasets in `ml/*/data/`):

```bash
python scripts/train_all_models.py
```

Or train individually:

```bash
python -m ml.gestational_diabetes.train
python -m ml.maternal_health.train
python -m ml.preeclampsia.train
```

Artifacts are saved to:

- `ml/gestational_diabetes/artifacts/`
- `ml/maternal_health/artifacts/`
- `ml/preeclampsia/artifacts/`

**Note:** Training uses GridSearchCV and may take 10–30 minutes depending on hardware.

### Legacy artifact preservation & migration

All model artifacts are properly organized into `ml/<model>/artifacts/`. The original historical implementations are preserved under `legacy/` (e.g., `legacy/Back-End/` and `legacy/Front-End/`), ensuring zero loss of past research and code.

The inference service `ml.predictor.ModelPredictor` natively checks `ml/<model>/artifacts/` first and includes backward-compatible fallbacks for `legacy/Back-End/` paths.

---

## Running the Application

### Flask web app (primary)

```bash
python run.py
```

Open [http://localhost:5000](http://localhost:5000)

### Streamlit demo (optional)

```bash
streamlit run streamlit_app/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Deployment

### Production checklist

1. Set `FLASK_ENV=production` and a strong `SECRET_KEY`
2. Set `SESSION_COOKIE_SECURE=true` (requires HTTPS)
3. Train models and verify artifacts exist
4. Use a production WSGI server:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 "run:app"
```

### Azure / cloud deployment

- Deploy behind HTTPS (Azure App Service, Container Apps, etc.)
- Mount persistent storage for `instance/database.db` or migrate to PostgreSQL
- Store `SECRET_KEY` in environment variables or a secrets manager
- Consider Azure Blob Storage for model artifacts in multi-instance setups

### Docker (suggested)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/train_all_models.py  # or copy pre-trained artifacts
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run:app"]
```

---

## Security Improvements (Recommended)

| Area | Current | Recommendation |
|------|---------|----------------|
| Password hashing | Werkzeug PBKDF2 | Consider `bcrypt` or `argon2` for production |
| CSRF protection | Not implemented | Add `Flask-WTF` on all POST forms |
| Rate limiting | None | Add `Flask-Limiter` on login/signup |
| Session storage | Client-side cookie | Use server-side sessions (Redis) at scale |
| Input validation | Basic range checks | Add schema validation (e.g. Marshmallow/Pydantic) |
| HTTPS | Header only | Enforce TLS at reverse proxy |
| Admin actions | Basic | Add role-based audit review dashboard |

---

## Performance Optimizations

- **Model caching:** `ModelPredictor` caches loaded artifacts in memory after first inference
- **Database:** Add indexes on `predictions(user_id)` and `predictions(timestamp)` for large histories
- **Static assets:** Serve via CDN or nginx in production
- **Training:** Reduce GridSearchCV parameter grid for faster iteration during development
- **PostgreSQL:** Replace SQLite when concurrent writes exceed ~100 users

---

## Model Performance

| Model | Algorithm | Imbalance Handling | Typical F1 |
|-------|-----------|-------------------|------------|
| GDM | XGBoost | SMOTE | ~0.87 |
| Maternal Health | XGBoost | BorderlineSMOTE | ~0.85 |
| Preeclampsia | XGBoost | SMOTE | ~0.86 |

---

## License

MIT License — see LICENSE file for details.

---

## Contributing

Contributions welcome. Please open an issue or pull request for bug fixes, UI improvements, or deployment automation.
