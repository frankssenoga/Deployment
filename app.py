import os
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

# Disable TensorFlow GPU initialization if TensorFlow is installed (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("ffnn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the feature names used during training
feature_names = [
    'cpu_utilization', 'memory_used', 'vdaerror', 'hdaerror'
    # ➕ Add your actual features from training here
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict using model
        prediction = model.predict(input_scaled)
        predicted_class = int(np.round(prediction[0]))

        return jsonify({
            'prediction': predicted_class,
            'raw_output': float(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run app using dynamic PORT (for Render or Heroku)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT; fallback to 5000 locally
    app.run(host='0.0.0.0', port=port)

