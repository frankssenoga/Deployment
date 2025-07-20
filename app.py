import os
import re
from flask import Flask, request, render_template, jsonify, abort
import joblib
import numpy as np
import lime
import lime.lime_tabular

app = Flask(__name__)

# ────────────────────────────────
# Load model and scaler
# ────────────────────────────────
MODEL_PATH = "ffnn_model_n.pkl"
SCALER_PATH = "scaler_n.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "rxbytes_rate", "txbytes_rate", "timecpu", "timesys", "timeusr",
    "state", "cputime", "memminor_fault", "memunused", "memlast_update",
    "memrss", "vdard_req_rate", "vdard_bytes_rate", "vdawr_reqs_rate",
    "vdawr_bytes_rate", "hdard_req_rate", "hdard_bytes_rate"
]

# ────────────────────────────────
# Utility: clean numeric values
# ────────────────────────────────
num_re = re.compile(r"[^\d\-.]")  # keep digits, dot, minus
def to_float(raw: str) -> float:
    cleaned = num_re.sub("", raw or "")
    if cleaned in ("", "-", ".", "-."):
        return 0.0
    return float(cleaned)

# ────────────────────────────────
# LIME explainer setup
# ────────────────────────────────
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.zeros((1, len(FEATURES))),  # Dummy shape
    mode='classification',
    feature_names=FEATURES,
    class_names=['Normal', 'Attack'],
    discretize_continuous=True
)

# ────────────────────────────────
# Home route
# ────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    empty_vals = {f: "" for f in FEATURES}
    return render_template("index.html", values=empty_vals, prediction=None, top_features=None)

# ────────────────────────────────
# Predict route
# ────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    is_json = request.is_json
    incoming = request.get_json(force=True) if is_json else request.form

    # Check for all required fields
    if not all(k in incoming for k in FEATURES):
        abort(400, description="Missing one or more required features.")

    # Convert input
    try:
        vals = [to_float(incoming[k]) for k in FEATURES]
    except ValueError as err:
        abort(400, description=f"Invalid numeric value: {err}")

    x_scaled = scaler.transform([vals])
    proba = model.predict_proba(x_scaled)[0][1]
    y_pred = int(proba >= 0.5)
    label = "🚨 Virtual Machine Under Attack" if y_pred else "✅ Virtual Machine Normal"

    # LIME explainability
    top_features = []
    try:
        explanation = explainer.explain_instance(
            np.array(vals), 
            lambda x: model.predict_proba(scaler.transform(x)), 
            num_features=5
        )
        top_features = explanation.as_list()
    except Exception as e:
        print("LIME explainability error:", str(e))

    # Prepare values for form display
    input_vals = {k: incoming[k] for k in FEATURES}

    if is_json:
        return jsonify({
            "prediction": y_pred,
            "probability": round(float(proba), 6),
            "explanation": dict(top_features)
        })

    return render_template(
        "index.html",
        values=input_vals,
        prediction=f"{label} (Prob={proba:.4f})",
        top_features=top_features
    )

# ────────────────────────────────
# Entry point
# ────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
