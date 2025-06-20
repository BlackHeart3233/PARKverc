import json
import os
import math

name_to_id = {
    "Ball_2025_03_30_20_54": 0,
    "Bicycle_2025_03_30_20_54": 1,
    "Bike_2025_03_30_20_54": 2,
    "Car_2025_03_30_20_54": 3,
    "Cart_2025_03_30_20_54": 4,
    "Curb_2025_03_30_20_54": 5,
    "Human_2025_03_30_20_54": 6,
    "Pole_2025_03_30_20_54": 7,
    "Ramp_2025_03_30_20_54": 8,
    "Parking_line_2025_03_30_20_54": 9,
    "Electric_car_2025_04_28_10_50": 10,
    "Family_car_2025_04_28_10_50": 11,
    "Handicapped_parking_2025_04_28_10_50": 12
}

videoName = "97bad2a6-za_Kjaro.mp4"
savePath = f"assets/Kjara/labels_image/{videoName[:-4]}/"
pathJSONMINI = "./assets/Kjara/labels_video/podatki_3_snemanje_7_4_8_9__zaKjaro3video.json"
with open(pathJSONMINI, "r") as file:
    data_all = json.load(file)

# Velikost slike - prilagodi po potrebi ali jo pridobi iz podatkov, če so
IMG_W = 1280
IMG_H = 720

# Poišči index videa
video_index = None
for idx, entry in enumerate(data_all):
    if videoName in entry["video"]:
        video_index = idx
        break
if video_index is None:
    raise ValueError(f"Video {videoName} not found in JSON")

os.makedirs(savePath, exist_ok=True)

data = data_all[video_index]
frame_count = data["box"][0]['framesCount']

# Ustvari prazne txt fajle za vse frame-e
for i in range(1, frame_count + 1):
    open(f"{savePath}frame_{i}.txt", "w").close()

def rotate_point(cx, cy, x, y, angle_rad):
    """Vrti točko (x,y) okoli centra (cx,cy) za angle_rad radianov."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_new = cx + cos_a * (x - cx) - sin_a * (y - cy)
    y_new = cy + sin_a * (x - cx) + cos_a * (y - cy)
    return x_new, y_new

for box in data["box"]:
    if not box['sequence']:
        continue
    if 'labels' not in box:
        print(f"Skipping box because 'labels' key is missing")
        continue

    class_label = box['labels'][0]
    class_id = name_to_id.get(class_label, None)
    if class_id is None:
        print(f"⚠️ Neznana oznaka: {class_label}, preskočena.")
        continue

    firstFrame = box['sequence'][0]['frame']
    lastFrame = box['sequence'][-1]['frame']

    seq_idx = 0
    current_frame = firstFrame

    for frame_num in range(firstFrame, lastFrame + 1):
        seq_item = box['sequence'][seq_idx]
        if frame_num > seq_item['frame'] and seq_idx + 1 < len(box['sequence']):
            seq_idx += 1
            seq_item = box['sequence'][seq_idx]

        x_pct = seq_item['x']  # v %
        y_pct = seq_item['y']
        w_pct = seq_item['width']
        h_pct = seq_item['height']
        angle_deg = seq_item.get('rotation', 0)
        enabled = seq_item.get('enabled', True)

        if not enabled:
            continue  # Čeobjekt ni viden v tem frame preskoči

        #Pretvori % v piksle
        abs_x = x_pct / 100 * IMG_W
        abs_y = y_pct / 100 * IMG_H
        abs_w = w_pct / 100 * IMG_W
        abs_h = h_pct / 100 * IMG_H

        #izračun centra
        cx = abs_x + abs_w / 2
        cy = abs_y + abs_h / 2

        #koordinate vogalov brez rotacije
        corners = [
            (abs_x, abs_y),  #top-left
            (abs_x + abs_w, abs_y),  #top-right
            (abs_x + abs_w, abs_y + abs_h),  #bottom-right
            (abs_x, abs_y + abs_h)  #bottom-left
        ]

        angle_rad = math.radians(angle_deg)

        #rotira vse točke okoli centra
        rotated_corners = [rotate_point(cx, cy, x, y, angle_rad) for (x, y) in corners]

        #Normaliziraj med 0 in 1
        norm_corners = [(x / IMG_W, y / IMG_H) for (x, y) in rotated_corners]

        #class_id x1 y1 x2 y2 x3 y3 x4 y4
        line = f"{class_id} " + " ".join(f"{coord:.6f}" for point in norm_corners for coord in point) + "\n"

        with open(f"{savePath}frame_{frame_num}.txt", "a") as f_out:
            f_out.write(line)

print(f"✅ OBB oznake so shranjene v {savePath}")
