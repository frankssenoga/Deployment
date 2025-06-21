from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('ffnn_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# List of features used in the model (from your notebook)
feature_names = [
    # Add your actual feature names here — extracted from X_train.columns
    'cpu_utilization', 'memory_used', 'vdaerror', 'hdaerror', 
    # Add the full list of columns as in df_cleaned.drop('Status', axis=1)
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        input_data = request.get_json()

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_class = int(np.round(prediction[0][0]))  # Assuming binary output

        return jsonify({
            'prediction': predicted_class,
            'probability': float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
