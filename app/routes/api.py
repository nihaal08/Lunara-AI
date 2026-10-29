"""API routes."""

from flask import Blueprint, jsonify, request

api_bp = Blueprint("api", __name__)


@api_bp.route("/api/theme", methods=["POST"])
def set_theme():
    data = request.get_json(silent=True) or {}
    theme = data.get("theme", "light")
    response = jsonify({"status": "success", "theme": theme})
    response.set_cookie("theme", theme, max_age=30 * 24 * 60 * 60)
    return response
