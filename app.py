"""
app.py
======
Main Flask application for "PCOS Early Screening and Awareness".
Processes questionnaire inputs purely in-memory, applies exact training preprocessing,
computes calibrated risk probabilities, generates SHAP explanations, and delivers
adaptive recommendations without persisting personal health data.
"""

import os
import logging
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

from utils.validator import validate_questionnaire_input, FIELD_SPECS
from utils.explainer import compute_shap_explanation
from utils.recommender import generate_recommendation_plans

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "pcos-early-screening-in-memory-secret")

# Suppress debug logging of inputs to ensure patient data privacy
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pcos_app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pcos_pipeline.joblib")

# ==========================================================
# LOAD ARTIFACTS ONCE AT STARTUP
# ==========================================================
PIPELINE_LOADED = False
MODEL_ARTIFACTS = None
MODEL_LOAD_ERROR = None

try:
    if os.path.exists(MODEL_PATH):
        MODEL_ARTIFACTS = joblib.load(MODEL_PATH)
        PIPELINE_LOADED = True
        logger.info("Pipeline and ML artifacts loaded successfully.")
    else:
        MODEL_LOAD_ERROR = f"Model pipeline artifact not found at {MODEL_PATH}. Please run train_model.py first."
        logger.error(MODEL_LOAD_ERROR)
except Exception as e:
    MODEL_LOAD_ERROR = f"Failed to initialize ML model pipeline: {str(e)}"
    logger.error(MODEL_LOAD_ERROR)


@app.context_processor
def inject_global_vars():
    """Injects model status and universal disclaimer."""
    return {
        'pipeline_loaded': PIPELINE_LOADED,
        'model_error': MODEL_LOAD_ERROR,
        'metrics': MODEL_ARTIFACTS.get('metrics', {}) if MODEL_ARTIFACTS else {},
        'universal_disclaimer': (
            "This prototype provides preliminary risk awareness only. "
            "It does not diagnose or rule out PCOS. Please consult a qualified healthcare professional."
        )
    }


# ==========================================================
# ROUTES
# ==========================================================

@app.route("/", methods=["GET"])
def index():
    """Home page with project overview, clinical disclaimer, and screening entrypoint."""
    return render_template("index.html")


@app.route("/screening", methods=["GET"])
def screening():
    """Questionnaire form for patient-facing risk screening."""
    if not PIPELINE_LOADED:
        flash("Screening system is temporarily unavailable because the ML model is not loaded. Please contact support or run train_model.py.", "danger")
        return render_template("index.html")

    feature_medians = MODEL_ARTIFACTS.get('feature_medians', {})
    return render_template("questionnaire.html", defaults=feature_medians, form_values={}, errors=[])


@app.route("/predict", methods=["POST"])
def predict():
    """
    Processes questionnaire inputs, runs calibrated prediction,
    computes SHAP explanation, and formats recommendation plan.
    Data is processed entirely in memory.
    """
    if not PIPELINE_LOADED:
        flash("Model artifact is missing or unreadable. Cannot generate screening estimate.", "danger")
        return redirect(url_for("index"))

    # Server-side validation
    feature_medians = MODEL_ARTIFACTS.get('feature_medians', {})
    sanitized_data, validation_errors = validate_questionnaire_input(request.form, feature_medians)

    if validation_errors:
        return render_template(
            "questionnaire.html",
            defaults=feature_medians,
            form_values=request.form,
            errors=validation_errors
        ), 400

    try:
        # Load components from in-memory artifacts
        calibrated_model = MODEL_ARTIFACTS['calibrated_model']
        base_xgb = MODEL_ARTIFACTS['base_xgb_model']
        preprocessor = MODEL_ARTIFACTS['preprocessor']
        feature_order = MODEL_ARTIFACTS['feature_order']
        display_names = MODEL_ARTIFACTS['display_names']
        thresholds = MODEL_ARTIFACTS['thresholds']

        # Construct single-row DataFrame in exact feature order
        input_df = pd.DataFrame([sanitized_data])[feature_order]

        # Preprocess input
        input_proc = preprocessor.transform(input_df)

        # Predict calibrated probability
        calibrated_prob = float(calibrated_model.predict_proba(input_proc)[:, 1][0])
        prob_percentage = round(calibrated_prob * 100, 1)

        # Map to data-driven provisional risk category
        if calibrated_prob < thresholds['low']:
            risk_category = "Low"
            risk_badge_class = "success"
        elif calibrated_prob < thresholds['high']:
            risk_category = "Moderate"
            risk_badge_class = "warning"
        else:
            risk_category = "High"
            risk_badge_class = "danger"

        # Compute SHAP explanation
        shap_data = compute_shap_explanation(
            base_xgb_model=base_xgb,
            preprocessor=preprocessor,
            feature_order=feature_order,
            display_names=display_names,
            user_dict=sanitized_data,
            top_k=5
        )

        # Generate recommendation plans
        rec_data = generate_recommendation_plans(
            risk_category=risk_category,
            user_data=sanitized_data
        )

        # Render results page (all data in memory, discarded after response)
        return render_template(
            "results.html",
            risk_category=risk_category,
            risk_badge_class=risk_badge_class,
            probability=calibrated_prob,
            probability_pct=prob_percentage,
            thresholds=thresholds,
            shap_data=shap_data,
            recommendations=rec_data,
            user_data=sanitized_data,
            display_names=display_names
        )

    except Exception as e:
        logger.error(f"Prediction error occurred: {str(e)}")
        # Safe, user-friendly error without stack traces
        return render_template(
            "questionnaire.html",
            defaults=feature_medians,
            form_values=request.form,
            errors=["An unexpected error occurred while processing your screening responses. Please verify your inputs and try again."]
        ), 500


# ==========================================================
# ERROR HANDLERS (No stack traces shown to users)
# ==========================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "index.html",
        custom_error="The requested page does not exist. Please return to the home page."
    ), 404


@app.errorhandler(500)
def server_error(e):
    return render_template(
        "index.html",
        custom_error="An internal error occurred. No personal data was saved. Please return to the home page."
    ), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
