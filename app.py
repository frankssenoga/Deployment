# app.py
import os
import re
from flask import Flask, request, render_template, jsonify, abort
import joblib
import numpy as np

from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# ────────────────────────────
# Load model & scaler once
# ────────────────────────────
MODEL_PATH  = "ffnn_model_n.pkl"
SCALER_PATH = "scaler_n.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ────────────────────────────
# Feature list (MUST match training order)
# ────────────────────────────
FEATURES = [
    "rxbytes_rate",  "txbytes_rate",  "timecpu",      "timesys",      "timeusr",
    "state",         "cputime",       "memminor_fault","memunused",    "memlast_update",
    "memrss",        "vdard_req_rate","vdard_bytes_rate","vdawr_reqs_rate",
    "vdawr_bytes_rate","hdard_req_rate","hdard_bytes_rate"
]

# ────────────────────────────
# Initialize LIME Explainer
# ────────────────────────────
explainer = LimeTabularExplainer(
    training_data=np.zeros((1, len(FEATURES))),  # dummy shape
    feature_names=FEATURES,
    class_names=["Normal", "Attack"],
    mode="classification",
    discretize_continuous=True
)

# ────────────────────────────
# Helper: clean and convert values
# ────────────────────────────
num_re = re.compile(r"[^\d\-.]")
def to_float(raw: str) -> float:
    cleaned = num_re.sub("", raw or "")
    if cleaned in ("", "-", ".", "-."):
        return 0.0
    return float(cleaned)

# ────────────────────────────
# Routes
# ────────────────────────────
@app.route("/", methods=["GET"])
def home():
    empty_vals = {f: "" for f in FEATURES}
    return render_template("index.html", values=empty_vals, prediction=None)

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

    x_scaled = scaler.transform([vals])
    proba    = model.predict_proba(x_scaled)[0][1]
    y_pred   = int(proba >= 0.5)

    if is_json:
        return jsonify({
            "prediction": y_pred,
            "probability": round(float(proba), 6)
        })

    # LIME Explainability Block
    try:
        # Define prediction function for LIME
        def predict_fn(x):
            return model.predict_proba(scaler.transform(x))

        lime_exp = explainer.explain_instance(
            np.array(vals),
            predict_fn,
            num_features=5
        )

        top_features = lime_exp.as_list()

    except Exception as e:
        top_features = [("Explainability Error", 0.0)]

    display_vals = {k: incoming[k] for k in FEATURES}
    label = "🚨 Virtual Machine Under Attack" if y_pred else "✅ Virtual Machine Normal"
    return render_template(
        "index.html",
        values=display_vals,
        prediction=f"{label} (Prob={proba:.4f})",
        top_features=top_features
    )

# ────────────────────────────
# Entrypoint
# ────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
