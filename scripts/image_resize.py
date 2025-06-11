import os
import shutil
from PIL import Image
from pathlib import Path
from typing import Optional
from configs.settings import (
    TARGET_IMAGE_SIZE,
    ORIGINAL_DATA_DIR,
    RESIZED_DATA_DIR,
    IMAGES_SUBDIR,
    LABELS_SUBDIR,
    CLASSES_FILENAME,
)


def resize_images(
    original_image_dir: Path,
    resized_image_dir: Path,
    target_size: int,
) -> None:
    """Resize all PNG images in the original directory and save them to the resized directory."""
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(resized_image_dir, exist_ok=True)
    processed = 0
    for img_path in original_image_dir.glob("*.png"):
        try:
            img = Image.open(img_path)
            img = img.convert("RGB")
            img.thumbnail((target_size, target_size))
            img.save(
                resized_image_dir / img_path.name,
                "PNG",
                quality=95,
                optimize=True,
                progressive=True,
            )
            processed += 1
            print(f"[INFO] Resized image: {img_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to process image {img_path}: {e}")
    print(f"✅ Resized {processed} images saved to: {resized_image_dir}")


def copy_labels(
    original_labels_dir: Path,
    resized_labels_dir: Path,
) -> None:
    """Copy label files from the original to the resized directory."""
    if original_labels_dir.exists():
        try:
            shutil.copytree(original_labels_dir, resized_labels_dir, dirs_exist_ok=True)
            print(f"✅ Labels copied to: {resized_labels_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to copy labels: {e}")
    else:
        print("⚠️ Labels folder not found.")


def copy_classes_file(
    classes_file: Path,
    resized_base_dir: Path,
    classes_filename: str,
) -> None:
    """Copy the classes.txt file to the resized directory."""
    dest = resized_base_dir / classes_filename
    if classes_file.exists():
        try:
            shutil.copy(classes_file, dest)
            print(f"✅ {classes_filename} copied to: {resized_base_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to copy {classes_filename}: {e}")
    else:
        print(f"⚠️ {classes_filename} not found.")


def prepare_resized_dataset() -> None:
    # === Config ===
    target_size: int = TARGET_IMAGE_SIZE
    original_base_dir: Path = Path(ORIGINAL_DATA_DIR)
    resized_base_dir: Path = Path(RESIZED_DATA_DIR)

    original_image_dir: Path = original_base_dir / IMAGES_SUBDIR
    resized_image_dir: Path = resized_base_dir / IMAGES_SUBDIR
    original_labels_dir: Path = original_base_dir / LABELS_SUBDIR
    resized_labels_dir: Path = resized_base_dir / LABELS_SUBDIR
    classes_file: Path = original_base_dir / CLASSES_FILENAME

    # === Resize images ===
    resize_images(original_image_dir, resized_image_dir, target_size)

    # === Copy labels ===
    copy_labels(original_labels_dir, resized_labels_dir)

    # === Copy classes.txt ===
    copy_classes_file(classes_file, resized_base_dir, CLASSES_FILENAME)


def main() -> None:
    prepare_resized_dataset()


if __name__ == "__main__":
    main()
