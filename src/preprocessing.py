# src/prepare_data.py
# Splits the raw images into data/train (80%) and data/test (20%).

import os
import shutil

from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def find_class_folder(root):
    """Find the folder that contains the 6 class folders (the zip is nested)."""
    for folder, dirs, _ in os.walk(root):
        if all(c in dirs for c in CLASSES):
            return folder
    raise FileNotFoundError("Could not find the 6 class folders inside data/raw")


def main():
    source = find_class_folder(RAW_DIR)
    for split_dir in (TRAIN_DIR, TEST_DIR):
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    for c in CLASSES:
        files = os.listdir(os.path.join(source, c))
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        for split_dir, split_files in ((TRAIN_DIR, train_files), (TEST_DIR, test_files)):
            os.makedirs(os.path.join(split_dir, c), exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(source, c, f), os.path.join(split_dir, c, f))
        print(f"{c}: {len(train_files)} train, {len(test_files)} test")


if __name__ == "__main__":
    main()