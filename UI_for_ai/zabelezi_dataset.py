import cv2
import os
import json
import numpy as np
from ultralytics import YOLO


labels_mini = {}
# === Naloži YOLO model ===
def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt"):
    return YOLO(url)

model = nalozi_model()

def obdelaj_sliko(frame, sigurnost=0.6):
    results = model(frame, verbose=False, conf=sigurnost)
    return results[0].plot(), results[0]

def izpisi_in_izlusci(result):
    oznake = []
    if result.obb and result.obb.cls is not None:
        for i, cls_id in enumerate(result.obb.cls.cpu().numpy()):
            label = model.names[int(cls_id)]
            coord = result.obb.xywhr.cpu().numpy()[i]
            oznake.append({
                "type": "obb",
                "label": label,
                "class_id": int(cls_id),
                "bbox": coord.tolist()
            })
    elif result.boxes and result.boxes.cls is not None:
        for i, cls_id in enumerate(result.boxes.cls.cpu().numpy()):
            label = model.names[int(cls_id)]
            coord = result.boxes.xyxy.cpu().numpy()[i]
            oznake.append({
                "type": "box",
                "label": label,
                "class_id": int(cls_id),
                "bbox": coord.tolist()
            })
    return oznake

def bezier_quad(p0, p1, p2, t):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def draw_bezier_curve(img, p0, p1, p2, color):
    points = []
    for t in np.linspace(0, 1, 100):
        pt = bezier_quad(np.array(p0), np.array(p1), np.array(p2), t)
        points.append(tuple(pt.astype(int)))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, 3)

def center_of_bbox(box):
    if len(box) == 5:  # OBB
        return (box[0], box[1])
    elif len(box) == 4:  # XYXY
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    return (0, 0)

def get_two_nearest(center, oznake):
    dists = []
    for oznaka in oznake:
        if oznaka['label'] == 'parking_line':
            c = center_of_bbox(oznaka['bbox'])
            dist = np.linalg.norm(np.array(center) - np.array(c))
            dists.append((dist, oznaka))
    dists.sort(key=lambda x: x[0])
    return [d[1] for d in dists[:2]]

# === Video posnetki ===
video_paths = [
    "IMG_4891.mp4",
    "IMG_4896.mp4",
    "IMG_4904.mp4",
    "IMG_4910.mp4"]

output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)
dataset_name = os.path.basename(output_dir)

offset = 0
step = 10
max_offset = 150
frame_global = 0
labels = {}

# === Procesiraj vsak video ===
for video_index, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    frame_local = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        start = (w // 2, h - 50)
        end = (w // 2 + offset, h // 3)
        control = ((start[0] + end[0]) // 2 + offset, (start[1] + end[1]) // 2)

        annotated_frame, result = obdelaj_sliko(frame)
        oznake = izpisi_in_izlusci(result)
        najblizji = get_two_nearest(center, oznake)

        display_frame = annotated_frame.copy()
        draw_bezier_curve(display_frame, start, control, end, (0, 255, 0))

        cv2.putText(display_frame, f"Dataset: {dataset_name}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Offset: {offset}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Video: {os.path.basename(video_path)}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)

        for i, ozn in enumerate(najblizji):
            cv2.putText(display_frame, f"Line {i+1}: {ozn['bbox']}", (30, 170 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        cv2.imshow("YOLO z označevanjem", display_frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            offset = max(offset - step, -max_offset)
        elif key == ord('d'):
            offset = min(offset + step, max_offset)
        elif key == ord('s'):
            filename = f"video{video_index}_frame_{frame_local:04d}.jpg"
            image_path = os.path.join(output_dir, filename)
            cv2.imwrite(image_path, frame)

            # Glavna oznaka
            labels[filename] = {
                "video": os.path.basename(video_path),
                "offset": offset,
                "curve": [list(start), list(control), list(end)],
                "nearest_lines": najblizji
            }

            # === Zbiranje mini oznak v en sam slovar ===
            labels_mini[filename] = {
                "offset": offset,
                "img_size": [frame.shape[1], frame.shape[0]],
                "detections": []
            }

            for ozn in najblizji:
                bbox = ozn["bbox"]
                center = center_of_bbox(bbox)
                labels_mini[filename]["detections"].append({
                    "bbox": bbox,
                    "center": [float(center[0]), float(center[1])]
                })

            print(f"[✔] Shranjeno: {filename} | offset={offset}")
            frame_local += 1
            frame_global += 1

        elif key == ord('q'):
                break

    cap.release()

cv2.destroyAllWindows()

# === Shrani vse oznake ===
with open(os.path.join(output_dir, "labels.json"), "w") as f:
    json.dump(labels, f, indent=4)

with open(os.path.join(output_dir, "labels_mini.json"), "w") as f:
    json.dump(labels_mini, f, indent=4)

