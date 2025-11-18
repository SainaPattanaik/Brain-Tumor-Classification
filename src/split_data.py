import os
import shutil
import random
from pathlib import Path

DATA_DIR = Path("data/train")
VAL_DIR = Path("data/val")
VAL_RATIO = 0.2  # 20%

VAL_DIR.mkdir(parents=True, exist_ok=True)

for cls in os.listdir(DATA_DIR):
    class_path = DATA_DIR / cls
    if not class_path.is_dir():
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]

    (VAL_DIR / cls).mkdir(parents=True, exist_ok=True)

    for img in val_images:
        src = class_path / img
        dest = VAL_DIR / cls / img
        shutil.move(src, dest)

print("Validation split created successfully!")