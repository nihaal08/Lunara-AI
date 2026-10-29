"""Admin dashboard routes."""

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

from app.decorators import admin_required

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    try:
        stats = {
            "total_users": current_app.auth_service.count_users(),
            "total_predictions": current_app.auth_service.count_predictions(),
            "active_users": current_app.auth_service.count_active_users(),
        }
        model_counts = current_app.prediction_service.count_by_model()
        users = current_app.auth_service.get_all_users()
        predictions = current_app.prediction_service.get_all_for_admin()

        return render_template(
            "admin.html",
            stats=stats,
            model_counts=model_counts,
            users=users,
            predictions=predictions,
        )
    except Exception:
        current_app.logger.exception("Admin dashboard error")
        flash("Error loading admin dashboard.", "error")
        return redirect(url_for("main.home"))


@admin_bp.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id: int):
    if user_id == session.get("user_id"):
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin.admin_dashboard"))

    try:
        current_app.auth_service.delete_user(user_id)
        current_app.audit_service.log(
            session["user_id"], "delete_user", {"deleted_user_id": user_id}
        )
        flash("User deleted successfully.", "success")
    except Exception:
        current_app.logger.exception("Delete user error")
        flash("Error deleting user.", "error")

    return redirect(url_for("admin.admin_dashboard"))
