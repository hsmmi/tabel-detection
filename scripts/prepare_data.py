import os
import shutil
from scripts.image_resize import prepare_resized_dataset
from scripts.split_dataset import split_resized_dataset
from configs.settings import RESIZED_DATA_DIR, RESIZED_DATA_SPLIT_DIR, SPLIT_RATIOS


def prepare_and_split_dataset():

    # Step 1: Resize the original images
    prepare_resized_dataset()

    # Step 2: Split the resized images
    split_resized_dataset(
        resized_dir=RESIZED_DATA_DIR,
        out_base=RESIZED_DATA_SPLIT_DIR,
        split_ratios=SPLIT_RATIOS,
    )

    # Step 3: Delete the resized images directory after splitting
    if os.path.exists(RESIZED_DATA_DIR):
        shutil.rmtree(RESIZED_DATA_DIR)


def main():
    prepare_and_split_dataset()


if __name__ == "__main__":
    main()
