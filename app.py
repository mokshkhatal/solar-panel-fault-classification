import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import torch

# Add src to sys.path so modules inside src can import each other without 'src.' prefix
sys.path.append(str(Path(__file__).parent / "src"))

# Import the prediction logic
from predict import predict_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("models/")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Save uploaded image
    file_path = Path(app.config['UPLOAD_FOLDER']) / file.filename
    file.save(file_path)
    
    try:
        # Run prediction
        pred_class, confidence, entropy, severity_level, delta_value = predict_image(
            image_path=file_path,
            models_dir=models_dir,
            device=device
        )
        
        # The grad-cam heatmap is hardcoded to save as prediction_heatmap.jpg in root
        heatmap_url = f"/heatmap?t={os.path.getmtime('prediction_heatmap.jpg')}"
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': pred_class,
                'confidence': float(confidence),
                'entropy': float(entropy),
                'severity': severity_level,
                'delta': float(delta_value)
            },
            'heatmap_url': heatmap_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap')
def heatmap():
    # Serve the generated heatmap
    return send_file('prediction_heatmap.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
