"""Flask application factory for LUNARA-AI."""

from datetime import timedelta

from flask import Flask, render_template, request, session

from app.config import get_config
from app.database import DatabaseManager
from app.logging_config import setup_logging
from app.routes import register_blueprints
from app.services.audit_service import AuditService
from app.services.auth_service import AuthService
from app.services.prediction_service import PredictionService
from ml.predictor import ModelPredictor


def create_app(config_class=None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    config = config_class or get_config()
    app.config.from_object(config)

    setup_logging(app)

    # Initialize shared services
    db_manager = DatabaseManager(app.config["DATABASE_PATH"])
    db_manager.init_db()

    app.db_manager = db_manager
    app.auth_service = AuthService(db_manager)
    app.prediction_service = PredictionService(db_manager)
    app.audit_service = AuditService(db_manager)
    app.model_predictor = ModelPredictor(app.config["MODELS_PATH"])

    register_blueprints(app)
    _register_middleware(app)
    _register_error_handlers(app)

    app.logger.info("LUNARA-AI application initialized")
    return app


def _register_middleware(app: Flask) -> None:
    @app.before_request
    def before_request():
        session.permanent = True
        app.permanent_session_lifetime = timedelta(hours=24)
        session.modified = True

    @app.after_request
    def set_security_headers(response):
        if not request.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' fonts.googleapis.com cdnjs.cloudflare.com; "
            "font-src 'self' fonts.gstatic.com cdnjs.cloudflare.com; "
            "img-src 'self' data: https:; connect-src 'self';"
        )
        return response


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(error):
        return render_template("error.html", error_code=404, message="Page not found"), 404

    @app.errorhandler(403)
    def forbidden(error):
        return render_template("error.html", error_code=403, message="Access forbidden"), 403

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.exception("Internal server error")
        return render_template("error.html", error_code=500, message="Internal server error"), 500
