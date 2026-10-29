import secrets
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


def get_or_create_csrf_token() -> str:
    """Generate or retrieve CSRF token for the session."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def _register_middleware(app: Flask) -> None:
    @app.context_processor
    def inject_template_globals():
        return {
            "csrf_token": get_or_create_csrf_token,
            "is_authenticated": "user_id" in session,
            "current_username": session.get("username"),
            "is_admin": bool(session.get("is_admin", False)),
        }

    @app.before_request
    def before_request():
        session.permanent = True
        app.permanent_session_lifetime = timedelta(hours=24)
        session.modified = True
        get_or_create_csrf_token()

        # Validate CSRF token on mutating requests
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if not app.config.get("TESTING", False):
                # Allow theme API if it uses JSON
                if not request.path.startswith("/api/"):
                    token = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
                    expected = session.get("csrf_token")
                    if not token or not secrets.compare_digest(token, expected or ""):
                        app.logger.warning("CSRF validation failed on %s from %s", request.path, request.remote_addr)
                        return (
                            render_template(
                                "error.html",
                                error_code=400,
                                message="Session validation failed (CSRF token mismatch). Please refresh the page and try again.",
                            ),
                            400,
                        )

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
    @app.errorhandler(400)
    def bad_request(error):
        return render_template("error.html", error_code=400, message="Bad Request"), 400

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
