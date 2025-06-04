import os
import shutil
import random
from pathlib import Path

# Nastavitve poti
images_root = Path("assets/Kjara/images_from_video")
labels_root = Path("assets/Kjara/labels_image")
output_base = Path("Stranski_model/split_train_everything")
train_split = 0.8

# Inicializacija map
for split in ["train", "val"]:
    (output_base / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_base / "labels" / split).mkdir(parents=True, exist_ok=True)

# Zberi vse slike rekurzivno
image_files = list(images_root.rglob("*.jpg"))
random.shuffle(image_files)

# Split
split_idx = int(len(image_files) * train_split)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Log manjkajočih label
missing_labels = []

def copy_files(file_list, split):
    for img_path in file_list:
        # Relativna pot (npr. Video_003_25_4_2025/frame_0001.jpg)
        rel_path = img_path.relative_to(images_root)
        label_path = labels_root / rel_path.with_suffix(".txt")

        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        # Ustvari ustrezne podmape
        (output_base / "images" / split / rel_path.parent).mkdir(parents=True, exist_ok=True)
        (output_base / "labels" / split / rel_path.parent).mkdir(parents=True, exist_ok=True)

        # Kopiraj sliko in label
        shutil.copy2(img_path, output_base / "images" / split / rel_path)
        shutil.copy2(label_path, output_base / "labels" / split / rel_path.with_suffix(".txt"))

# Kopiranje
copy_files(train_files, "train")
copy_files(val_files, "val")

# YAML
yaml_path = output_base / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"""path: {output_base}
train: images/train
val: images/val
names:
  0: Žoga                   
  1: Kolo                   
  2: Motorno_kolo            
  3: Avtomobil               
  4: Nakupovalni_vozicek     
  5: Robnik                  
  6: Človek               
  7: Steber                  
  8: Rampa                  
  9: Prosto_parkirno_mesto   
  10: Drevo                  
  11: Prehod_za_pešce       
  12: Električno_parkiranje 
  13: Družinsko_parkiranje  
  14: Invalidsko_parkiranje  
  15: Prepovedano_parkiranje 
  16: Zasebno_parkirišče     
""")


# Log manjkajočih label
if missing_labels:
    log_path = output_base / "missing_labels.log"
    with open(log_path, "w") as f:
        f.write("\n".join(missing_labels))
    print(f"⚠️ Manjkajoče oznake (labels) zapisane v: {log_path}")

print("✅ Dataset pripravljen za učenje v:", output_base)
