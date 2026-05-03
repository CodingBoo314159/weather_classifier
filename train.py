import numpy as np
from ultralytics import YOLO
# Load pretrained model with 20 epochs
model = YOLO('yolov8n-cls.pt')
model.train(
    data=r"C:\Users\Damia\OneDrive\Desktop\Boon Projects\Boon Datasets\weather_dataset",
    epochs=20,
    imgsz=64
)
# Make Predictions 
# Load Model 
model = YOLO(r"C:\Users\Damia\Boon Kaggle\runs\classify\train-3\weights\last.pt")

# Predict using an image 
results = model(r"C:\Users\Damia\OneDrive\Pictures\Screenshots\Screenshot 2026-05-03 194115.png")

names_dict=results[0].names
probs=results[0].probs.data.tolist()

print(names_dict[np.argmax(probs)])
