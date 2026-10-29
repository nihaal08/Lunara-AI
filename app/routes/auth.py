"""Authentication routes."""

import sqlite3

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("main.home"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        if not identifier or not password:
            flash("Please provide both username/email and password.", "error")
            return render_template("login.html")

        try:
            user = current_app.auth_service.authenticate(identifier, password)
            if user:
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["email"] = user["email"]
                session["is_admin"] = user["is_admin"]
                current_app.audit_service.log(user["id"], "login")
                flash(f"Welcome back, {user['username']}!", "success")
                return redirect(url_for("main.home"))
            flash("Invalid username/email or password.", "error")
        except Exception:
            current_app.logger.exception("Login error")
            flash("An error occurred during login. Please try again.", "error")

    return render_template("login.html")


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("main.home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not all([username, full_name, email, password, confirm_password]):
            flash("All fields are required.", "error")
            return render_template("signup.html")
        if len(username) < 3:
            flash("Username must be at least 3 characters long.", "error")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("signup.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        try:
            current_app.auth_service.create_user(username, full_name, email, password)
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("auth.login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
        except Exception:
            current_app.logger.exception("Signup error")
            flash("An error occurred during registration.", "error")

    return render_template("signup.html")


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    user_id = session.get("user_id")
    if user_id:
        current_app.audit_service.log(user_id, "logout")
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("main.home"))
