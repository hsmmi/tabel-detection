import random
import shutil
from pathlib import Path
from typing import List, Dict
from configs.settings import RESIZED_DATA_DIR, RESIZED_DATA_SPLIT_DIR, SPLIT_RATIOS


def split_filenames(
    all_files: List[str], ratios: Dict[str, float]
) -> Dict[str, List[str]]:
    total_count = len(all_files)
    split_counts = {split: int(total_count * ratio) for split, ratio in ratios.items()}
    # Adjust last split count to account for rounding errors
    assigned = sum(split_counts.values())
    if assigned < total_count:
        last_split = list(ratios.keys())[-1]
        split_counts[last_split] += total_count - assigned

    split_data: Dict[str, List[str]] = {}
    start = 0
    for split, count in split_counts.items():
        split_data[split] = all_files[start : start + count]
        start += count
    return split_data


def copy_split_files(
    files: List[str],
    split: str,
    source_img_dir: Path,
    source_lbl_dir: Path,
    out_base: Path,
) -> None:
    for folder in ["images", "labels"]:
        (out_base / folder / split).mkdir(parents=True, exist_ok=True)

    for name in files:
        src_img = source_img_dir / f"{name}.png"
        dst_img = out_base / "images" / split / f"{name}.png"
        if not src_img.exists():
            print(f"⚠️ Warning: Image file {src_img} does not exist. Skipping.")
        else:
            try:
                shutil.copy(src_img, dst_img)
            except Exception as e:
                print(f"❌ Failed to copy {src_img} to {dst_img}: {e}")

        src_lbl = source_lbl_dir / f"{name}.txt"
        dst_lbl = out_base / "labels" / split / f"{name}.txt"
        if not src_lbl.exists():
            print(f"⚠️ Warning: Label file {src_lbl} does not exist. Skipping.")
        else:
            try:
                shutil.copy(src_lbl, dst_lbl)
            except Exception as e:
                print(f"❌ Failed to copy {src_lbl} to {dst_lbl}: {e}")


def copy_classes_file(source_lbl_dir: Path, out_base: Path) -> None:
    classes_src = source_lbl_dir.parent / "classes.txt"
    classes_dst = out_base / "classes.txt"
    if not classes_src.exists():
        print(f"⚠️ Warning: classes.txt file {classes_src} does not exist.")
    else:
        try:
            shutil.copy(classes_src, classes_dst)
        except Exception as e:
            print(f"❌ Failed to copy {classes_src} to {classes_dst}: {e}")


def split_resized_dataset(
    resized_dir: Path, out_base: Path, split_ratios: Dict[str, float]
) -> None:
    # Ensure resized_dir and out_base are Path objects
    resized_dir = Path(resized_dir)
    out_base = Path(out_base)
    random.seed(42)
    source_img_dir = resized_dir / "images"
    source_lbl_dir = resized_dir / "labels"
    all_files: List[str] = [f.stem for f in source_img_dir.glob("*.png")]
    random.shuffle(all_files)

    split_data = split_filenames(all_files, split_ratios)

    for split, files in split_data.items():
        copy_split_files(files, split, source_img_dir, source_lbl_dir, out_base)

    copy_classes_file(source_lbl_dir, out_base)

    print("✅ Dataset split complete.")


def main():
    resized_dir = RESIZED_DATA_DIR
    split_output_dir = RESIZED_DATA_SPLIT_DIR
    split_ratios = SPLIT_RATIOS
    split_resized_dataset(resized_dir, split_output_dir, split_ratios)


if __name__ == "__main__":
    main()
