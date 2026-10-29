"""ML prediction routes."""

import pandas as pd
from flask import Blueprint, current_app, flash, render_template, request, session

from app.decorators import login_required

predictions_bp = Blueprint("predictions", __name__)


@predictions_bp.route("/predict/gdm", methods=["GET", "POST"])
@login_required
def predict_gdm():
    result = None
    inputs = None

    if request.method == "POST":
        try:
            inputs = {
                "Age": float(request.form.get("age", 0)),
                "No of Pregnancy": int(request.form.get("no_pregnancy", 0)),
                "BMI": float(request.form.get("bmi", 0)),
                "HDL": float(request.form.get("hdl", 0)),
                "Family History": int(request.form.get("family_history", 0)),
                "Large Child or Birth Default": int(request.form.get("large_child", 0)),
                "PCOS": int(request.form.get("pcos", 0)),
                "Sys_BP": float(request.form.get("sys_bp", 0)),
                "Dia_BP": float(request.form.get("dia_bp", 0)),
                "OGTT": float(request.form.get("ogtt", 0)),
                "Sedentary Lifestyle": int(request.form.get("sedentary", 0)),
                "Prediabetes": int(request.form.get("prediabetes", 0)),
            }

            if not (18 <= inputs["Age"] <= 60):
                flash("Age must be between 18 and 60.", "error")
                return render_template("predict_gdm.html")
            if not (0 <= inputs["No of Pregnancy"] <= 15):
                flash("Number of pregnancies must be between 0 and 15.", "error")
                return render_template("predict_gdm.html")
            if not (15 <= inputs["BMI"] <= 55):
                flash("BMI must be between 15 and 55.", "error")
                return render_template("predict_gdm.html")

            result = current_app.model_predictor.predict_gdm(pd.DataFrame([inputs]))
            current_app.prediction_service.save(
                session["user_id"], "Gestational Diabetes", inputs, result
            )
            current_app.audit_service.log(
                session["user_id"], "predict_gdm", {"prediction": result}
            )
            flash("Prediction completed successfully!", "success")
        except ValueError:
            flash("Invalid input. Please check your values.", "error")
        except Exception as exc:
            current_app.logger.exception("GDM prediction error")
            flash(f"Error during prediction: {exc}", "error")

    return render_template("predict_gdm.html", result=result, inputs=inputs)


@predictions_bp.route("/predict/maternal", methods=["GET", "POST"])
@login_required
def predict_maternal():
    result = None
    inputs = None

    if request.method == "POST":
        try:
            inputs = {
                "Age": float(request.form.get("age", 0)),
                "SystolicBP": float(request.form.get("systolic_bp", 0)),
                "DiastolicBP": float(request.form.get("diastolic_bp", 0)),
                "BS": float(request.form.get("bs", 0)),
                "HeartRate": int(request.form.get("heart_rate", 0)),
            }

            if not (15 <= inputs["Age"] <= 60):
                flash("Age must be between 15 and 60.", "error")
                return render_template("predict_maternal.html")

            result = current_app.model_predictor.predict_maternal(pd.DataFrame([inputs]))
            current_app.prediction_service.save(
                session["user_id"], "Maternal Health Risk", inputs, result
            )
            current_app.audit_service.log(
                session["user_id"], "predict_maternal", {"prediction": result}
            )
            flash("Prediction completed successfully!", "success")
        except ValueError:
            flash("Invalid input. Please check your values.", "error")
        except Exception as exc:
            current_app.logger.exception("Maternal prediction error")
            flash(f"Error during prediction: {exc}", "error")

    return render_template("predict_maternal.html", result=result, inputs=inputs)


@predictions_bp.route("/predict/preeclampsia", methods=["GET", "POST"])
@login_required
def predict_preeclampsia():
    result = None
    inputs = None

    if request.method == "POST":
        try:
            inputs = {
                "gravida": int(request.form.get("gravida", 0)),
                "parity": int(request.form.get("parity", 0)),
                "Gestational_Age_Weeks": float(request.form.get("gestational_age", 0)),
                "Age": float(request.form.get("age", 0)),
                "BMI": float(request.form.get("bmi", 0)),
                "diabetes": int(request.form.get("diabetes", 0)),
                "Hypertension_History": int(request.form.get("hypertension", 0)),
                "Systolic BP": float(request.form.get("systolic_bp", 0)),
                "Diastolic BP": float(request.form.get("diastolic_bp", 0)),
                "HB": float(request.form.get("hb", 0)),
                "Fetal_Weight": float(request.form.get("fetal_weight", 0)),
                "Protein_Uria": int(request.form.get("protein_uria", 0)),
                "Amniotic_Fluid_Levels": float(request.form.get("amniotic_fluid", 0)),
            }

            if not (15 <= inputs["Age"] <= 60):
                flash("Age must be between 15 and 60.", "error")
                return render_template("predict_preeclampsia.html")

            result = current_app.model_predictor.predict_preeclampsia(pd.DataFrame([inputs]))
            current_app.prediction_service.save(
                session["user_id"], "Preeclampsia Risk", inputs, result
            )
            current_app.audit_service.log(
                session["user_id"], "predict_preeclampsia", {"prediction": result}
            )
            flash("Prediction completed successfully!", "success")
        except ValueError:
            flash("Invalid input. Please check your values.", "error")
        except Exception as exc:
            current_app.logger.exception("Preeclampsia prediction error")
            flash(f"Error during prediction: {exc}", "error")

    return render_template("predict_preeclampsia.html", result=result, inputs=inputs)
