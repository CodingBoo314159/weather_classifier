import numpy as np
from ultralytics import YOLO

# Load pretrained model and train
model = YOLO('yolov8n-cls.pt')
model.train(
    data="path/to/your/weather_dataset",  # update this path
    epochs=20,
    imgsz=64
)

# Load trained model and make predictions
model = YOLO("path/to/your/weights/last.pt")  # update this path

# Predict using an image
results = model("path/to/your/image.png")  # update this path
names_dict = results[0].names
probs = results[0].probs.data.tolist()
print(names_dict[np.argmax(probs)])
