import os
import re
from flask import Flask, request, render_template, abort
import joblib
import numpy as np
import shap

app = Flask(__name__)

# ──────────── Load model and scaler ────────────
MODEL_PATH  = "ffnn_model_n.pkl"
SCALER_PATH = "scaler_n.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ──────────── Define feature order ────────────
FEATURES = [
    "rxbytes_rate", "txbytes_rate", "timecpu", "timesys", "timeusr",
    "state", "cputime", "memminor_fault", "memunused", "memlast_update",
    "memrss", "vdard_req_rate", "vdard_bytes_rate", "vdawr_reqs_rate",
    "vdawr_bytes_rate", "hdard_req_rate", "hdard_bytes_rate"
]

# ──────────── Helper to clean numeric inputs ────────────
num_re = re.compile(r"[^\d\-.]")  # Keep digits, dot, minus
def to_float(raw: str) -> float:
    cleaned = num_re.sub("", raw or "")
    if cleaned in ("", "-", ".", "-."):
        return 0.0
    return float(cleaned)

# ──────────── Home Route ────────────
@app.route("/", methods=["GET"])
def home():
    empty_vals = {f: "" for f in FEATURES}
    return render_template("index.html", values=empty_vals, prediction=None, top_features=[])

# ──────────── Prediction Route ────────────
@app.route("/predict", methods=["POST"])
def predict():
    is_json = request.is_json
    incoming = request.get_json(force=True) if is_json else request.form

    # Ensure all required fields are present
    if not all(k in incoming for k in FEATURES):
        abort(400, description="Missing one or more required features.")

    # Parse and scale inputs
    try:
        vals = [to_float(incoming[k]) for k in FEATURES]
    except ValueError as err:
        abort(400, description=f"Bad numeric value → {err}")

    x_scaled = scaler.transform([vals])
    proba = model.predict_proba(x_scaled)[0][1]
    y_pred = int(proba >= 0.5)

    # ──────────── SHAP Explainability ────────────
    try:
        background = x_scaled  # Use the input row as lightweight background
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(x_scaled, nsamples=50)

        # Use class 1 SHAP values
        feature_contributions = shap_values[1][0]
        contributions = list(zip(FEATURES, feature_contributions))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    except Exception as e:
        print("SHAP explainability error:", e)
        top_features = [("Explainability Error", str(e))]

    # Render result
    display_vals = {k: incoming[k] for k in FEATURES}
    label = "🚨 Virtual Machine Under Attack" if y_pred else "✅ Virtual Machine Normal"

    return render_template(
        "index.html",
        values=display_vals,
        prediction=f"{label} (Prob={proba:.4f})",
        top_features=top_features
    )

# ──────────── Run locally ────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
