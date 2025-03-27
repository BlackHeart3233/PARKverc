import os
import shutil
import random

# 🔧 Prilagodi poti glede na tvojo strukturo
images_dir = 'images'
labels_dir = 'labels'
output_dir = 'datasets/dataset'

split_ratio = 0.8  # 80% train, 20% val
splits = ['train', 'val']

# Dobimo vse slike (podpira .jpg, .jpeg, .png)
images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)
split_index = int(len(images) * split_ratio)

for i, split in enumerate(splits):
    img_split = images[:split_index] if split == 'train' else images[split_index:]

    for img_file in img_split:
        # Ime label datoteke (.txt)
        label_file = os.path.splitext(img_file)[0] + '.txt'

        # Ustvari potrebne mape
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

        # Kopiraj sliko
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_dir, 'images', split, img_file)
        )

        # Kopiraj label, če obstaja
        label_src = os.path.join(labels_dir, label_file)
        label_dst = os.path.join(output_dir, 'labels', split, label_file)

        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"⚠️  Opozorilo: Label datoteka za {img_file} ne obstaja.")
