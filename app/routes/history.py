"""Prediction history and PDF export routes."""

from datetime import datetime

from flask import Blueprint, current_app, flash, make_response, redirect, render_template, send_file, session, url_for

from app.decorators import login_required
from app.services.pdf_service import generate_prediction_pdf

history_bp = Blueprint("history", __name__)


@history_bp.route("/history")
@login_required
def view_history():
    try:
        predictions = current_app.prediction_service.get_user_history(session["user_id"])
        return render_template("history.html", predictions=predictions)
    except Exception:
        current_app.logger.exception("History load error")
        flash("Error loading prediction history.", "error")
        return redirect(url_for("main.home"))


@history_bp.route("/download-history", methods=["POST"])
@login_required
def download_history():
    try:
        predictions = current_app.prediction_service.get_user_history(session["user_id"])
        pdf_buffer = generate_prediction_pdf(predictions)
        return make_response(
            send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=f"LUNARA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mimetype="application/pdf",
            )
        )
    except Exception:
        current_app.logger.exception("PDF generation error")
        flash("Error generating PDF report.", "error")
        return redirect(url_for("history.view_history"))


@history_bp.route("/clear-history", methods=["POST"])
@login_required
def clear_history():
    try:
        current_app.prediction_service.clear_user_history(session["user_id"])
        current_app.audit_service.log(session["user_id"], "clear_history")
        flash("Prediction history cleared successfully.", "success")
    except Exception:
        current_app.logger.exception("Clear history error")
        flash("Error clearing history.", "error")
    return redirect(url_for("history.view_history"))
