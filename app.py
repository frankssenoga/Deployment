# app.py
import os
import re
from flask import Flask, request, render_template, jsonify, abort
import joblib
import numpy as np

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
num_re = re.compile(r"[^\d\-.]")       # keep 0‑9, dot, minus
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
    # blank defaults so the form loads empty
    empty_vals = {f: "" for f in FEATURES}
    return render_template("index.html", values=empty_vals, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts:
      • a browser form (application/x-www-form-urlencoded)
      • or JSON: {"rxbytes_rate": 123, ...}
    Returns:
      • HTML (if form)   — renders index.html with prediction
      • JSON (if JSON)   — {"prediction": 1, "probability": 0.987}
    """
    # decide if JSON or form
    is_json = request.is_json

    incoming = request.get_json(force=True) if is_json else request.form

    # make sure all features are present
    if not all(k in incoming for k in FEATURES):
        abort(400, description="Missing one or more required features.")

    # convert to float list in correct order
    try:
        vals = [to_float(incoming[k]) for k in FEATURES]
    except ValueError as err:
        abort(400, description=f"Bad numeric value → {err}")

    x_scaled = scaler.transform([vals])
    proba    = model.predict_proba(x_scaled)[0][1]
    y_pred   = int(proba >= 0.5)

    if is_json:   # return JSON
        return jsonify({
            "prediction": y_pred,
            "probability": round(float(proba), 6)
        })

    # else render HTML
    display_vals = {k: incoming[k] for k in FEATURES}
    label = f"{'🚨 Virtual Machine Under Attack' if y_pred else '✅ Virtual Machine Normal'} ({proba:.1%})"
    return render_template(
        "index.html",
        values=display_vals,
        prediction=f"{label} (Prob={proba:.4f})"
    )

# ────────────────────────────
# Entry‑point
# ────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
