import os
import shutil
import random
from pathlib import Path

# Paths
images_dir = Path("assets/Kjara/images_from_video/Video_007_25_4_2025")
labels_dir = Path("assets/Kjara/labels_image/Video_007_25_4_2025_MINI_JSON")
output_base = Path("assets/Kjara/yolo_data")
train_split = 0.8

for split in ["train", "val"]:
    os.makedirs(output_base / "images" / split, exist_ok=True)
    os.makedirs(output_base / "labels" / split, exist_ok=True)

image_files = sorted(images_dir.glob("*.jpg"))
random.shuffle(image_files)

split_idx = int(len(image_files) * train_split)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def copy_files(file_list, split):
    for img_path in file_list:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue  
        shutil.copy2(img_path, output_base / "images" / split / img_path.name)
        shutil.copy2(label_path, output_base / "labels" / split / label_path.name)

copy_files(train_files, "train")
copy_files(val_files, "val")

yaml_path = output_base / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {output_base}
train: images/train
val: images/val
names:
  0: Ball_2025_03_30_20_54
  1: Bicycle_2025_03_30_20_54
  2: Bike_2025_03_30_20_54
  3: Car_2025_03_30_20_54
  4: Cart_2025_03_30_20_54
  5: Curb_2025_03_30_20_54
  6: Human_2025_03_30_20_54
  7: Pole_2025_03_30_20_54
  8: Ramp_2025_03_30_20_54
  9: Parking_line_2025_03_30_20_54
  10: Electric_car_2025_04_28_10_50
  11: Family_car_2025_04_28_10_50
  12: Handicapped_parking_2025_04_28_10_50
""")

print("✅ Dataset prepared for training in:", output_base)
print("👉 To train, run:\n\n  yolo detect train model=yolov8n.pt data=assets/Kjara/yolo_data/data.yaml epochs=50 imgsz=640")
