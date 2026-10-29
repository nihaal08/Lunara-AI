"""Register all application blueprints."""

from app.routes.admin import admin_bp
from app.routes.api import api_bp
from app.routes.auth import auth_bp
from app.routes.history import history_bp
from app.routes.main import main_bp
from app.routes.predictions import predictions_bp


def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
