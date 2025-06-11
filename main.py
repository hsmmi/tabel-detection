import os
from ultralytics import YOLO
from configs.settings import MODEL_PATH, DATA_CONFIG, TARGET_IMAGE_SIZE, PROJECT_NAME
from scripts.create_yolo_config import create_yolo_config


def get_last_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("epoch") and f.endswith(".pt")
        ],
        key=lambda x: int(x.replace("epoch", "").replace(".pt", "")),
    )
    return os.path.join(checkpoint_dir, checkpoints[-1]) if checkpoints else None


YOLO_MODEL_PATH = MODEL_PATH  # Path to the YOLOv12 model weights
DATA_CONFIG = DATA_CONFIG
IMAGE_SIZE = TARGET_IMAGE_SIZE

create_yolo_config(DATA_CONFIG)

# YOLOv12 small model
# Check if there is a previous checkpoint to resume from
checkpoint_dir = os.path.join(PROJECT_NAME, "weights")

last_checkpoint = get_last_checkpoint(checkpoint_dir)

if last_checkpoint:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    model = YOLO(last_checkpoint)
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
    resume=bool(last_checkpoint),
)
