import os
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib


# Tell any installed TensorFlow to stay on CPU and keep logs quiet
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # 0=all, 1=info, 2=warning, 3=error

# ────────────────────────────────────────────────────────────
# 2. Load model and scaler
# ────────────────────────────────────────────────────────────
MODEL_PATH  = "ffnn_model_fn.pkl"   # <-- make sure this file is in the same folder
SCALER_PATH = "scaler_fn.pkl"       # <-- ditto (rename if you kept scaler_n.pkl)

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ────────────────────────────────────────────────────────────
# 3. Feature list (exactly the 17 columns used in training)
# ────────────────────────────────────────────────────────────
feature_names = [
    "rxbytes_rate",    "txbytes_rate",
    "timecpu",         "timesys",        "timeusr",
    "state",           "cputime",
    "memminor_fault",  "memunused",      "memlast_update",
    "memrss",
    "vdard_req_rate",  "vdard_bytes_rate",
    "vdawr_reqs_rate", "vdawr_bytes_rate",
    "hdard_req_rate",  "hdard_bytes_rate"
]

# ────────────────────────────────────────────────────────────
# 4. Flask application
# ────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON body containing every feature in `feature_names`.
    Returns the binary class and the raw model score/probability.
    """
    try:
        payload = request.get_json(force=True)

        # Put payload into a single-row DataFrame in the correct order
        X = pd.DataFrame([payload], columns=feature_names)

        # Scale → Predict
        X_scaled      = scaler.transform(X)
        proba         = float(model.predict(X_scaled)[0])      # raw sigmoid output
        predicted_cls = int(round(proba))                      # 0 or 1

        return jsonify({
            "prediction": predicted_cls,
            "probability": proba
        })

    except Exception as exc:
        # Return the error message so you can debug client-side
        return jsonify({"error": str(exc)}), 400


# ────────────────────────────────────────────────────────────
# 5. Local / Render entry-point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Render (and Heroku) inject the desired port via $PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
