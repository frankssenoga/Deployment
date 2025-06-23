from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("ffnn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the feature names used during training (MUST match your training set)
feature_names = [
    'cpu_utilization', 'memory_used', 'vdaerror', 'hdaerror',
    # 👉 Add the full and exact list of features used in model training
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        input_data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Scale the data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # If classifier: round or use predict_proba
        predicted_class = int(np.round(prediction[0]))

        return jsonify({
            'prediction': predicted_class,
            'raw_output': float(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
