import os
from ultralytics import YOLO
from configs.settings import MODEL_PATH, DATA_CONFIG, TARGET_IMAGE_SIZE
from scripts.create_yolo_config import create_yolo_config

YOLO_MODEL_PATH = MODEL_PATH  # Path to the YOLOv12 model weights
DATA_CONFIG = DATA_CONFIG
IMAGE_SIZE = TARGET_IMAGE_SIZE

create_yolo_config(DATA_CONFIG)

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
