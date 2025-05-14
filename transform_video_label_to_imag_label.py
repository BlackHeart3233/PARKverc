import json
import os

# 🗂️ Nastavi poti
input_json = "assets/Kjara/labels_video/Video_007_25_4_2025_MINI_JSON.json"
output_dir = "assets/Kjara/labels_image/Video_007_25_4_2025_MINI_JSON"

# 📂 Ustvari izhodno mapo, če še ne obstaja
os.makedirs(output_dir, exist_ok=True)

# 📖 Naloži JSON datoteko
with open(input_json, "r") as f:
    data = json.load(f)

# 📦 Shrani YOLO oznake za vsak frame
frame_annotations = {}

for obj in data[0]["box"]:
    for seq in obj["sequence"]:
        if not seq.get("enabled", False):
            continue
        frame = seq["frame"]
        x = seq["x"]
        y = seq["y"]
        w = seq["width"]
        h = seq["height"]

        # 🎯 Pretvori v YOLO format (normalizirano)
        x_center = (x + w / 2) / 100
        y_center = (y + h / 2) / 100
        w_norm = w / 100
        h_norm = h / 100

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        frame_annotations.setdefault(frame, []).append(yolo_line)

# 📝 Zapiši frame_X.txt datoteke
for frame_num in range(1, 367):  # 366 framov
    lines = frame_annotations.get(frame_num, [])
    output_path = os.path.join(output_dir, f"frame_{frame_num}.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
