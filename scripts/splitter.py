import os
import random
import shutil

train_dir = "data/train"
val_dir = "data/val"
split_fraction = 0.25

os.makedirs(val_dir, exist_ok=True)

# List all image files in train_dir
image_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpeg'))]

# Randomly select 
num_val = int(split_fraction * len(image_files))
val_files = random.sample(image_files, num_val)

# Move selected files to val_dir
for fname in val_files:
    src = os.path.join(train_dir, fname)
    dst = os.path.join(val_dir, fname)
    shutil.move(src, dst)

print(f"Moved {len(val_files)} files from train to val folder.")