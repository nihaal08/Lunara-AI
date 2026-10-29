"""Public page routes (home, about, contact)."""

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return redirect(url_for("main.home"))


@main_bp.route("/home")
def home():
    return render_template(
        "home.html",
        username=session.get("username"),
        is_authenticated="user_id" in session,
    )


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thank you for your message. We will get back to you soon!", "success")
        return redirect(url_for("main.contact"))
    return render_template("contact.html")
