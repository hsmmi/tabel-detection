import random
import shutil
from pathlib import Path

random.seed(42)

# config
source_img_dir = Path("yolov12-tabel-detection/data/all_resized_1024x1024/images")
source_lbl_dir = Path("yolov12-tabel-detection/data/all_resized_1024x1024/labels")
out_base = Path("yolov12-tabel-detection/data")

# collect all base filenames (no extension)
all_files = [f.stem for f in source_img_dir.glob("*.png")]
random.shuffle(all_files)
data_count = len(all_files)

split_counts = {
    "train": int(data_count * 0.8),
    "val": int(data_count * 0.15),
    "test": data_count - int(data_count * 0.8) - int(data_count * 0.15),
}

# split them
split_data = {}
start = 0
for split, count in split_counts.items():
    split_data[split] = all_files[start : start + count]
    start += count

# move files
for split, files in split_data.items():
    for folder in ["images", "labels"]:
        (out_base / folder / split).mkdir(parents=True, exist_ok=True)

    for name in files:
        shutil.copy(
            source_img_dir / f"{name}.png", out_base / "images" / split / f"{name}.png"
        )
        shutil.copy(
            source_lbl_dir / f"{name}.txt", out_base / "labels" / split / f"{name}.txt"
        )

print("✅ Dataset split complete.")
