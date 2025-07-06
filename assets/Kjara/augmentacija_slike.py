
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
from pathlib import Path
import argparse

def transform(slika_path: Path):
    image = plt.imread(slika_path)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    scenarij = random.randint(0, 2)
    if scenarij == 0:
        transform = A.Compose([
            A.CLAHE(clip_limit=8.0, tile_grid_size=(8, 8), p=1.0),
            A.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.5), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4),
                                       contrast_limit=(0.6, 1.0), p=1.0)
        ])
    elif scenarij == 1:
        transform = A.Compose([
            A.RandomFog(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(2, 5), p=0.2)
        ])
    else:
        transform = A.Compose([
            A.ImageCompression(quality_range=(10, 15), p=1.0),
        ])
    return transform(image=image)["image"]

def procesiraj_slikovno_mapo(input_dir: Path): #augmentira vsako sliko 1× in shrani v <ime>_augmentirano"
    output_dir = input_dir.parent / f"{input_dir.name}_augmentirano"
    output_dir.mkdir(parents=True, exist_ok=True)

    slikovne_pripone = {".jpg", ".jpeg", ".png"}
    slikovne_datoteke = [p for p in input_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in slikovne_pripone]

    if not slikovne_datoteke:
        print(f"{input_dir}: brez slik preskočim.")
        return

    for slika in slikovne_datoteke:
        aug_slika = transform(slika)
        plt.imsave(output_dir / slika.name, aug_slika)

    print(f"Augmentiral {len(slikovne_datoteke)} slik → {output_dir}")

def kopiraj_label_mapo(label_dir: Path): #kopira celotno mapo lablov v <ime>_augmentirano"
    output_dir = label_dir.parent / f"{label_dir.name}_augmentirano"
    shutil.copytree(label_dir, output_dir, dirs_exist_ok=True)
    print(f"Kopiral labele → {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Augmentacija slik (1×) in kopiranje label map.")
    parser.add_argument("--images_root", default="images",
                        help="Koren mape s podmapami slik")
    parser.add_argument("--labels_root", default="labels",
                        help="Koren mape s podmapami label datotek")
    args = parser.parse_args()

    images_root  = Path(args.images_root)
    labels_root  = Path(args.labels_root)

    if not images_root.is_dir():
        raise SystemExit(f"Mapa {images_root} ne obstaja.")
    if not labels_root.is_dir():
        raise SystemExit(f"Mapa {labels_root} ne obstaja.")

    podmape_slik = [p for p in images_root.iterdir() if p.is_dir()]
    if not podmape_slik:
        raise SystemExit(f"Mapa {images_root} nima podmap.")

    print(f"Najdene podmape slik: {[p.name for p in podmape_slik]}")

    for img_dir in podmape_slik:
        procesiraj_slikovno_mapo(img_dir)

        lbl_dir = labels_root / img_dir.name
        if lbl_dir.is_dir():
            kopiraj_label_mapo(lbl_dir)
        else:
            print(f"Manjka label mapa za {img_dir.name} – preskoci.")

if __name__ == "__main__":
    main()
