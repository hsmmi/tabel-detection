
# Table Detection with YOLOv12

This project focuses on detecting tables in images using the YOLOv12 object detection architecture. It involves resizing large images, splitting datasets for training, validation, and testing, and training a custom YOLO model.

## 📁 Project Structure

```
tabel-detection/
├── configs/
│   └── settings.py               # Configuration constants (paths, image size, etc.)
├── data/
│   ├── all_original/            # Raw images and labels
│   ├── all_resized_640x640/     # Resized images
│   └── split/                   # Split dataset (train/val/test)
├── models/
│   └── yolov12s.pt              # Pretrained YOLOv12 model
├── outputs/
│   └── table-run-python/        # YOLO training outputs
├── scripts/
│   ├── image_resize.py          # Resize high-resolution images
│   ├── split_dataset.py         # Split dataset into train/val/test
│   └── prepare_data.py          # Run both resize and split together
├── .env                         # PYTHONPATH and environment setup
└── README.md                    # Project documentation
```

## ⚙️ Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/tabel-detection.git
   cd tabel-detection
   ```

2. Install dependencies with [uv](https://github.com/astral-sh/uv):

   ```bash
   uv pip install -r requirements.txt
   ```

3. Make sure the `.env` file sets:

   ```env
   PYTHONPATH=.
   ```

4. Download the pretrained YOLOv12 model and place it in `models/yolov12s.pt`.

## 🛠️ Preprocessing Pipeline

Run the following to resize and split your data:

```bash
python scripts/prepare_data.py
```

This will:

- Resize original images to 640x640 (or your configured size)
- Save them under `data/all_resized_640x640/`
- Split into train/val/test and save under `data/split/`

## 🚀 Training the Model (Python API)

```python
from ultralytics import YOLO
from configs.settings import YOLO_MODEL_PATH, YOLO_DATA_CONFIG_PATH, TARGET_IMAGE_SIZE

model = YOLO(YOLO_MODEL_PATH)
model.train(
    data=YOLO_DATA_CONFIG_PATH,
    epochs=30,
    imgsz=TARGET_IMAGE_SIZE,
    batch=8,
    project="outputs",
    name="table-run-python",
    pretrained=True,
    verbose=True,
)
```
