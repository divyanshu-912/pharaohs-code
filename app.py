from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import sys
import tensorflow as tf

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from backend.models.captcha_model import CaptchaModel
from backend.utils.image_processing import preprocess_image
from backend.config import Config

app = Flask(__name__)
CORS(app)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Load the trained model
try:
    model = CaptchaModel()
    model_path = os.path.join(current_dir, 'models', 'captcha_model.keras')
    if os.path.exists(model_path):
        model.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/api/decode', methods=['POST'])
def decode_captcha():
    if not model:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image provided'
        }), 400
    
    try:
        image = request.files['image']
        
        # Validate file type
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({
                'success': False,
                'error': 'Invalid file format. Please upload PNG or JPG images.'
            }), 400

        # Process image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        decoded_text = model.decode_prediction(prediction)
        
        # Calculate confidence scores
        confidence_scores = np.max(prediction, axis=-1)[0]
        avg_confidence = float(np.mean(confidence_scores))
        
        return jsonify({
            'success': True,
            'result': decoded_text,
            'confidence': avg_confidence,
            'confidence_per_char': [float(score) for score in confidence_scores]
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting server on port {port}")
    print(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
