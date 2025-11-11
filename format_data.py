import os
import shutil
from glob import glob
import yaml

# This file is for formatting the FER2013 dataset into YOLO format for training for detection.

# Define file paths
source_dir = "FER2013"
target_dir = "dataset"

# Define emotion classes
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Create YOLO directory structure
for split in ["train", "val"]:
    os.makedirs(os.path.join(target_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, split, "labels"), exist_ok=True)

def process_split(split_name, source_split):
    for class_id, emotion in enumerate(classes):
        emotion_dir = os.path.join(source_split, emotion)
        images = glob(os.path.join(emotion_dir, "*.jpg")) + glob(os.path.join(emotion_dir, "*.png"))

        for img_path in images:
            # Copy image
            img_name = os.path.basename(img_path)
            dest_img_path = os.path.join(target_dir, split_name, "images", img_name)
            shutil.copy(img_path, dest_img_path)

            # Create YOLO label file
            label_path = os.path.join(target_dir, split_name, "labels", os.path.splitext(img_name)[0] + ".txt")
            with open(label_path, "w") as f:
                # Full image = one bounding box
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Process train and test as val
process_split("train", os.path.join(source_dir, "train"))
process_split("val", os.path.join(source_dir, "test"))

# Create data.yaml
yaml_path = os.path.join(target_dir, "data.yaml")
data = {
    "train": "dataset/train/images",
    "val": "dataset/val/images",
    "nc": len(classes),
    "names": classes
}
with open(yaml_path, "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print("Dataset formatted successfully!")
