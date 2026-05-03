from flask import Flask, render_template, request
from ultralytics import YOLO
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Important: stops matplotlib from opening a window
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Config ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ← add this line
MODEL_PATH = 'last.pt'
RESULTS_CSV = 'your_results.csv'

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

    # 3. Generate training graphs and save as images
    df = pd.read_csv(RESULTS_CSV)

    def save_plot(x, y, title, ylabel, filename):
        plt.figure()
        plt.plot(x, y)
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.savefig(f'static/{filename}')
        plt.close()

    save_plot(df['epoch'], df['train/loss'], 'Train Loss', 'Loss', 'train_loss.png')
    save_plot(df['epoch'], df['val/loss'], 'Validation Loss', 'Loss', 'val_loss.png')
    save_plot(df['epoch'], df['metrics/accuracy_top1'] * 100, 'Top-1 Accuracy', 'Accuracy (%)', 'accuracy.png')

    return render_template('results.html',
        label=predicted_label,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)
