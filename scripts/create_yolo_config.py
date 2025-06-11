from configs.settings import RESIZED_DATA_SPLIT_DIR
import yaml
from pathlib import Path


def create_yolo_config(config_path: Path):
    data_yaml = {
        "path": str(RESIZED_DATA_SPLIT_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["table"],
    }

    with open(config_path, "w") as f:
        yaml.dump(data_yaml, f)


# Usage
create_yolo_config(Path("configs/yolov12_custom.yaml"))
