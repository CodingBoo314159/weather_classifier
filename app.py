from flask import Flask, render_template, request
from ultralytics import YOLO
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Config ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = 'last.pt'

model = YOLO(MODEL_PATH)

# --- Home page ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Save uploaded image
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # 2. Run model
    results = model(filepath)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    predicted_label = names_dict[np.argmax(probs)]
    confidence = round(max(probs) * 100, 2)

    return render_template('results.html',
        label=predicted_label,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
