import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL = "yolov12s.pt"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, YOLO_MODEL)
DATA_CONFIG = os.path.join(BASE_DIR, "configs", "yolov12_custom.yaml")
IMAGE_SIZE = 1024  # Set the image size for training

# YOLOv12 small model
model = YOLO(YOLO_MODEL_PATH)

# Train with custom config
model.train(
    data=DATA_CONFIG,
    epochs=30,
    imgsz=IMAGE_SIZE,
    batch=8,
    project="outputs",
    name="table-run-python",
    pretrained=True,
    verbose=True,
)
