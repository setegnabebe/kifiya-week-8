# serve_model.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained model
with open('model/rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Log incoming request
    logging.info("Received request for prediction")

    # Get JSON data
    data = request.get_json()
    if not data:
        logging.error("No data provided")
        return jsonify({"error": "No data provided"}), 400
    
    try:
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data).tolist()  # For probabilities
        
        # Log prediction result
        logging.info(f"Prediction: {prediction[0]}, Probability: {prediction_proba[0]}")

        # Send response
        return jsonify({
            "prediction": int(prediction[0]),
            "prediction_proba": prediction_proba[0]
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
