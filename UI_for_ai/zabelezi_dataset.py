import cv2
import os
import json
import numpy as np

video_path = "Video_004_28_3_2025.mp4"
output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_num = 0
offset = 0  # začetni offset
step = 10
max_offset = 150

labels = {}

# Bezier funkcija (kvadratna)
def bezier_quad(p0, p1, p2, t):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

# Funkcija za risanje krivulje
def draw_bezier_curve(img, p0, p1, p2, color):
    points = []
    for t in np.linspace(0, 1, 100):
        pt = bezier_quad(np.array(p0), np.array(p1), np.array(p2), t)
        points.append(tuple(pt.astype(int)))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, 3)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    h, w = frame.shape[:2]

    # Točke za Bezier
    start = (w // 2, h - 50)  # center spodaj
    end = (w // 2 + offset, h // 3)  # končna točka se rahlo zamakne z offsetom
    control = ((start[0] + end[0]) // 2 + offset, (start[1] + end[1]) // 2)

    draw_bezier_curve(display_frame, start, control, end, (0, 255, 0))  # zelena krivulja

    # Prikaži offset na zaslonu
    cv2.putText(display_frame, f"Offset: {offset}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Označevanje + krivulja", display_frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('a'):
        offset = max(offset - step, -max_offset)
    elif key == ord('d'):
        offset = min(offset + step, max_offset)
    elif key == ord('s'):
        # shrani frame in offset
        frame_filename = f"frame_{frame_num:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
        labels[frame_filename] = offset
        print(f"[✔] Shranjeno: {frame_filename} -> {offset}")
        frame_num += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# shrani oznake
with open(os.path.join(output_dir, "labels.json"), "w") as f:
    json.dump(labels, f, indent=4)
