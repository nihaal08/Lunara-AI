"""Route access control decorators."""

from functools import wraps

from flask import redirect, render_template, session, url_for


def login_required(view):
    """Require an authenticated user session."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    """Require an authenticated admin user."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id") or not session.get("is_admin"):
            return (
                render_template(
                    "error.html",
                    error_code=403,
                    message="Unauthorized - Admin access required",
                ),
                403,
            )
        return view(*args, **kwargs)

    return wrapped
