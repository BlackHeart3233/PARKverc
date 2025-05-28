import os
import cv2
import glob
import random
import numpy as np
from pathlib import Path
import albumentations as A

INPUT_IMAGE_DIR = "raw/images"
INPUT_LABEL_DIR = "raw/labels"
OUTPUT_IMAGE_DIR = "augmented/images"
OUTPUT_LABEL_DIR = "augmented/labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def clip(v): return max(min(v, 1.0), 0.0)

# flip x-coordinates of the polygon
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
        # naključna svetlost
        factor = random.randint(-40, 40)
        # naključna kontrast
        hsv = cv2.cvtColor(img_aug, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], factor)

        img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # blur
    if random.random() < 0.3:
        img_aug = cv2.GaussianBlur(img_aug, (3, 3), 0)

    # šum
    if random.random() < 0.3:
        noise = np.random.normal(1, 3, img_aug.shape).astype(np.uint8)
        img_aug = cv2.add(img_aug, noise)

    return img_aug, labels_aug

def enhance_parking_lines(image):
    enhance_transform = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Sharpen(alpha=(0.6, 1.0), lightness=(1.0, 1.4), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.4, p=1.0)
    ])
    return enhance_transform(image=image)['image']


"""
augmentacija na eni sliki
:param signal: vhodna slika
:param labels: seznam oznak v formatu [(class, x1, y1, ..., x4, y4), ...]
:return: slovar z augmentiranimi podatki
"""
def transform(signal, labels=None):
    img_aug, labels_aug = apply_augmentation(signal, labels or [])

    img_aug = enhance_parking_lines(img_aug)

    result = {
        "image": img_aug,
        "labels": labels_aug
    }
    return result


def main():
    # get all images
    image_files = glob.glob(f"{INPUT_IMAGE_DIR}/*.jpg")

    # validacija
    if len(image_files) == 0:
        print("ni najdenih vhodnih slik.")
        return

    # slika in index
    for idx, img_path in enumerate(image_files[:10]):

        # pridobi ime slike brez končnice
        base = Path(img_path).stem
        # pridobi path labele
        label_path = os.path.join(INPUT_LABEL_DIR, f"{base}.txt")

        # preveri ali obstaja labela
        if not os.path.exists(label_path):
            print(f"ni labele za {base}")
            continue

        # slika
        img = cv2.imread(img_path)

        with open(label_path, "r") as f:
            labels = []
            for line in f:
                # preberi vrstico in pretvori v seznam
                parts = line.strip().split()

                # ime razreda in koordinate
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))

                # preveri dolžino koord
                if len(coords) == 8:
                    labels.append((cls, *coords))
                else:
                    print(f"neveljaven format: {label_path}")
                    continue

        transformed = transform(img, labels)

        # pridobi augmentirano sliko in oznake
        img_aug = transformed["image"]
        labels_aug = transformed["labels"]

        # shranjevanje
        new_name = f"{base}_demo"
        out_img_path = os.path.join(OUTPUT_IMAGE_DIR, f"{new_name}.jpg")
        out_lbl_path = os.path.join(OUTPUT_LABEL_DIR, f"{new_name}.txt")

        # shranjevanje slik in oznak
        cv2.imwrite(out_img_path, img_aug)

        with open(out_lbl_path, "w") as f:
            # ime in koordinate
            for cls, *coords in labels_aug:

                # omeji koordinate med 0 in 1
                clipped = [clip(c) for c in coords]

                #  zapiši ime + koordinate
                f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in clipped) + "\n")

    print("zaključeno -> 10 augmentiranih slik shranjenih")


if __name__ == "__main__":
    main()
