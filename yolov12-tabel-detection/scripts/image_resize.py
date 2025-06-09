import os
from PIL import Image
from pathlib import Path

target_size = 1024
orginal_image_dir = Path("yolov12-tabel-detection/data/all_original/images")
resize_image_dir = Path(
    f"yolov12-tabel-detection/data/all_resized_{target_size}x{target_size}/images"
)
os.makedirs(resize_image_dir, exist_ok=True)

for img_path in orginal_image_dir.glob("*.png"):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img.thumbnail((target_size, target_size))
    img.save(
        resize_image_dir / img_path.name,
        "PNG",
        quality=95,
        optimize=True,
        progressive=True,
    )
