import os
import cv2
import glob
import random
import numpy as np
from pathlib import Path

INPUT_IMAGE_DIR = "raw/images"
INPUT_LABEL_DIR = "raw/labels"
OUTPUT_IMAGE_DIR = "augmented/images"
OUTPUT_LABEL_DIR = "augmented/labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def clip(v): return max(min(v, 1.0), 0.0)

def flip_polygon(polygon):
    flipped = []
    for i in range(0, len(polygon), 2):
        flipped.append(1.0 - polygon[i])  # flip x
        flipped.append(polygon[i + 1])    # keep y
    return flipped

def apply_augmentation(img, labels):
    h_img, w_img = img.shape[:2]
    labels_aug = labels.copy()
    img_aug = img.copy()

    # horizontalni obrat
    if random.random() < 0.5:
        img_aug = cv2.flip(img_aug, 1)
        labels_aug = [(cls, *flip_polygon(points)) for cls, *points in labels_aug]

    # svetlost
    if random.random() < 0.5:
        factor = random.randint(-40, 40)
        hsv = cv2.cvtColor(img_aug, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], factor)
        img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # blur
    if random.random() < 0.3:
        img_aug = cv2.GaussianBlur(img_aug, (5, 5), 0)

    # šum
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img_aug.shape).astype(np.uint8)
        img_aug = cv2.add(img_aug, noise)

    return img_aug, labels_aug

# naloži slike
image_files = glob.glob(f"{INPUT_IMAGE_DIR}/*.jpg")

for img_path in image_files:
    base = Path(img_path).stem
    label_path = os.path.join(INPUT_LABEL_DIR, f"{base}.txt")

    if not os.path.exists(label_path):
        print(f"⚠️ ni labele za {base}")
        continue

    img = cv2.imread(img_path)

    with open(label_path, "r") as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            if len(coords) == 8:  # OBB format
                labels.append((cls, *coords))
            else:
                print(f"neveljaven format: {label_path}")
                continue

    for i in range(1, 6):  # 5 augmentacij
        img_aug, labels_aug = apply_augmentation(img, labels)

        new_name = f"{base}_aug{i}"
        out_img_path = os.path.join(OUTPUT_IMAGE_DIR, f"{new_name}.jpg")
        out_lbl_path = os.path.join(OUTPUT_LABEL_DIR, f"{new_name}.txt")

        cv2.imwrite(out_img_path, img_aug)

        with open(out_lbl_path, "w") as f:
            for cls, *coords in labels_aug:
                clipped = [clip(c) for c in coords]
                f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in clipped) + "\n")

print("končano.")
