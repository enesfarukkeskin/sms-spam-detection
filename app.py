"""
Flask application for SMS Spam Detection API.
"""

from flask import Flask, request, render_template, jsonify
from src.model import SMSSpamDetector
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = SMSSpamDetector()
    model.load_model('models/trained_model.pkl')
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get message from request
        if request.is_json:
            # API request
            data = request.get_json()
            message = data.get('message', '')
        else:
            # Form request
            message = request.form.get('message', '')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Make prediction
        prediction = model.predict([message])[0]
        confidence = 0.95  # You can add confidence score calculation here

        result = {
            'message': message,
            'prediction': prediction,
            'confidence': confidence
        }

        # Return based on request type
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', result=result)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        error_message = {'error': 'Error processing request'}
        return jsonify(error_message), 500

@app.route('/api/docs')
def api_docs():
    return render_template('api_docs.html')

if __name__ == '__main__':
    if model is None:
        logger.error("Could not start application - model not loaded")
    else:
        app.run(debug=True, host='0.0.0.0', port=5001)