TARGET_IMAGE_SIZE = 640

# Directory paths
ORIGINAL_DATA_DIR = "data/all_original"
RESIZED_DATA_DIR = f"data/all_resized_{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}"
RESIZED_DATA_SPLIT_DIR = f"data/split_resized_{TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}"
IMAGES_SUBDIR = "images"
LABELS_SUBDIR = "labels"
CLASSES_FILENAME = "classes.txt"
SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.15,
    "test": 0.05,
}
# Training config
EPOCHS = 30
BATCH_SIZE = 8
MODEL_NAME = "yolov8n.pt"
PROJECT_NAME = "runs"
EXPERIMENT_NAME = "table-run-python"
