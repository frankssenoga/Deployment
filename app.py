import os
import pandas as pd
import re
import joblib
import numpy as np
import shap
from flask import Flask, request, render_template, jsonify, abort

app = Flask(__name__)

# Load model & scaler
MODEL_PATH  = "ffnn_model_n.pkl"
SCALER_PATH = "scaler_n.pkl"
model       = joblib.load(MODEL_PATH)
scaler      = joblib.load(SCALER_PATH)

# Feature names (must match training)
FEATURES = [
    "rxbytes_rate", "txbytes_rate", "timecpu", "timesys", "timeusr",
    "state", "cputime", "memminor_fault", "memunused", "memlast_update",
    "memrss", "vdard_req_rate", "vdard_bytes_rate", "vdawr_reqs_rate",
    "vdawr_bytes_rate", "hdard_req_rate", "hdard_bytes_rate"
]

# Helper: Clean numeric input
num_re = re.compile(r"[^\d\-.]")
def to_float(raw: str) -> float:
    cleaned = num_re.sub("", raw or "")
    return float(cleaned) if cleaned not in ("", "-", ".", "-.") else 0.0

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", values={f: "" for f in FEATURES}, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    is_json = request.is_json
    incoming = request.get_json(force=True) if is_json else request.form

    if not all(k in incoming for k in FEATURES):
        abort(400, description="Missing one or more required features.")

    try:
        vals = [to_float(incoming[k]) for k in FEATURES]
    except ValueError as err:
        abort(400, description=f"Bad numeric value → {err}")

    df_input = pd.DataFrame([vals], columns=FEATURES)

    # Apply scaler and predict
    x_scaled = scaler.transform(df_input)
    proba = model.predict_proba(x_scaled)[0][1]
    y_pred   = int(proba >= 0.5)

    # SHAP explainability
    def explain_with_shap(model, scaled_input, background=None, feature_names=None):
        try:
            import shap
            # Use background of 100 mean samples if not provided
            if background is None:
                background = shap.kmeans(scaled_input, 1)

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(scaled_input)

            # If binary classifier, shap_values is a list [class_0, class_1]
            importances = shap_values[1][0]  # class 1, first (only) instance

            top_indices = np.argsort(np.abs(importances))[::-1][:12]

            top_features = []
            for i in top_indices:
                top_features.append({
                    "feature": feature_names[i],
                    "value": round(float(scaled_input[0, i]), 4),
                    "impact": round(float(importances[i]), 6)
                })
            return top_features

        except Exception as e:
            print("SHAP explainability error:", str(e))
            return []

    # Get SHAP explanations
    background = np.zeros((1, len(FEATURES)))  # zero background
    top_features = explain_with_shap(model, x_scaled, background, FEATURES)

    if is_json:
        return jsonify({
            "prediction": y_pred,
            "probability": round(float(proba), 6),
            "top_features": top_features
        })

    display_vals = {k: incoming[k] for k in FEATURES}
    # Convert probability to percentage
    percentage = proba * 100
    label = f"{'🚨 Virtual Machine Under Attack' if y_pred else '✅ Virtual Machine Normal'} (Confidence: {percentage:.1f}%)"
    return render_template(
        "index.html",
        values=display_vals,
        prediction=label,
        top_features=top_features
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
