import os
import re
from flask import Flask, request, render_template, jsonify, abort
import joblib
import numpy as np
import shap

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
# Helper: robust numeric cleaner
# ────────────────────────────
num_re = re.compile(r"[^\d\-.]")  # keep digits, dot, minus
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
    return render_template("index.html", values=empty_vals, prediction=None, top_features=[])

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
    proba = model.predict_proba(x_scaled)[0][1]
    y_pred = int(proba >= 0.5)

    # SHAP explainability
    try:
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], shap.kmeans(x_scaled, 1))
        shap_values = explainer.shap_values(x_scaled)
        contributions = list(zip(FEATURES, shap_values[0]))
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    except Exception as e:
        top_features = [("Explainability Error", str(e))]

    display_vals = {k: incoming[k] for k in FEATURES}
    label = "🚨 Virtual Machine Under Attack" if y_pred else "✅ Virtual Machine Normal"

    # 🔽 THIS IS THE CORRECT LOCATION FOR return
    return render_template(
        "index.html",
        values=display_vals,
        prediction=f"{label} (Prob={proba:.4f})",
        top_features=top_features  # ← Correctly included
    )

# ────────────────────────────
# Entry‑point
# ────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
