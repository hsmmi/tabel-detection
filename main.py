import os
from ultralytics import YOLO
from configs.settings import MODEL_PATH, DATA_CONFIG, TARGET_IMAGE_SIZE, PROJECT_NAME
from scripts.create_yolo_config import create_yolo_config

YOLO_MODEL_PATH = MODEL_PATH  # Path to the YOLOv12 model weights
DATA_CONFIG = DATA_CONFIG
IMAGE_SIZE = TARGET_IMAGE_SIZE

create_yolo_config(DATA_CONFIG)

# YOLOv12 small model
# Check if there is a previous checkpoint to resume from
checkpoint_dir = os.path.join(PROJECT_NAME, "weights")
if os.path.exists(checkpoint_dir):
    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("epoch") and f.endswith(".pt")
        ],
        key=lambda x: int(x.replace("epoch", "").replace(".pt", "")),
    )
    if checkpoints:
        last_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        model = YOLO(last_checkpoint)
    else:
        model = YOLO(YOLO_MODEL_PATH)
else:
    model = YOLO(YOLO_MODEL_PATH)

# Train with custom config
model.train(
    data=DATA_CONFIG,
    epochs=30,
    imgsz=IMAGE_SIZE,
    batch=8,
    project=PROJECT_NAME,
    name="table-run-python",
    pretrained=True,
    verbose=True,
    save_period=1,
    exist_ok=True,
)
