from ultralytics import YOLO
import cv2
from pathlib import Path

custom_imena = {
    0: 'Žoga',
    1: 'Kolo',
    2: 'Motor',
    3: 'Avto',
    4: 'Voziček',
    5: 'Robnik',
    6: 'Človek',
    7: 'Parkirna črta',
    8: 'Drog',
    9: 'Ovira',
    10: 'Drevo/Grm',
    11: 'Zebra'
}


def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt"):
    """naloži YOLOv8 model iz podane poti ali URL-ja."""
    model = YOLO(url)
    return model

path = Path("../model_1/runs/obb/train7/weights/best.pt")
print("📂 Absolutna pot:", path.resolve())
print("✅ Obstaja?", path.exists())

model = nalozi_model(str(path))

PARKING_LINE_CLASS_ID = 7
ZEBRA_CLASS_ID = 11

def obdelaj_sliko(frame, sigurnost = 0.6, debugger_mode = False):
    """
    obdelaj posamezen okvir (sliko) in vrni:
    - označen okvir (annotated image)
    - rezultate detekcije
    """
    results = model(frame, verbose=False, conf=sigurnost)
    results[0].names = custom_imena
    annotated = results[0].plot()

    if not debugger_mode:
        slika = pripravi_rezultat(frame, results)
        return slika, results[0], izracunaj_ovire(results[0])
    else:
        return annotated, results[0], izracunaj_ovire(results[0])

import numpy as np
import cv2

# zaenkrat samo rišemo črte

def pripravi_rezultat(frame, results):
    frame_copy = frame.copy()

    if hasattr(results[0], 'obb') and results[0].obb is not None:
        for i, obb_data_row in enumerate(results[0].obb.xywhr):
            cls_id = results[0].obb.cls[i].item()
            if cls_id == PARKING_LINE_CLASS_ID:
                x, y, w, h, angle = obb_data_row.tolist()

                if w <= 0 or h <= 0:
                    continue

                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                w2 = w / 2
                h2 = h / 2

                corners = np.array([
                    [-w2, -h2],
                    [ w2, -h2],
                    [ w2,  h2],
                    [-w2,  h2]
                ])

                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])

                pts = np.int32(rotated_corners).reshape((-1, 1, 2))

                hull = cv2.convexHull(pts)
                cv2.drawContours(frame_copy, [hull], 0, (0, 255, 0), 2)

    return frame_copy


def izracunaj_ovire(results) -> int:
    if not hasattr(results, 'obb') or results.obb is None or results.obb.xywhr is None or len(results.obb.xywhr) == 0:
        return 0

    image_width = results.orig_shape[1]
    image_height = results.orig_shape[0]

    left_boundary = image_width * 0.35
    right_boundary = image_width * 0.65
    bottom_threshold = image_height * 0.9

    danger_level = 0

    for i, obb_data_row in enumerate(results.obb.xywhr):
        current_class_id = results.obb.cls[i].item()
        if current_class_id == PARKING_LINE_CLASS_ID or current_class_id == ZEBRA_CLASS_ID:
            continue

        x_center = obb_data_row[0].item()
        y_center = obb_data_row[1].item()
        height = obb_data_row[3].item()  # višina iz xywhr: [x, y, w, h, r]

        bottom_edge = y_center + height / 2

        if left_boundary <= x_center <= right_boundary:
            if bottom_edge >= bottom_threshold:
                danger_level = max(danger_level, 3)
            else:
                danger_level = max(danger_level, 2)

    return danger_level

def izpisi_in_izlusci(result):
    oznake = []

    obb = result.obb
    if obb is not None and obb.cls is not None and obb.xywhr is not None:
        cls_ids = obb.cls.cpu().numpy()
        coords = obb.xywhr.cpu().numpy()
        confs = obb.conf.cpu().numpy() if obb.conf is not None else [-1.0]*len(cls_ids)

        #print(f" Detektiranih OBB objektov: {len(cls_ids)}")
        for i, cls_id in enumerate(cls_ids):
            label = model.names[int(cls_id)]
            coord = coords[i]
            conf = confs[i] if i < len(confs) else -1
           # print(f"OBB:  Razred: {label}, Koordinate: {coord}, Confidence: {conf:.2f}")

            oznake.append({
                "type": "obb",
                "label": label,
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": coord.tolist()
            })

    elif result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.cpu().numpy()
        coords = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [-1.0]*len(class_ids)

        #print(f" Detektiranih standardnih boxov: {len(class_ids)}")
        for i, cls_id in enumerate(class_ids):
            label = model.names[int(cls_id)]
            coord = coords[i]
            conf = confs[i] if i < len(confs) else -1
            #print(f"Box:  Razred: {label}, Koordinate: {coord}, Confidence: {conf:.2f}")

            oznake.append({
                "type": "box",
                "label": label,
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": coord.tolist()
            })

    else:
        print(" Ni zaznanih oznak ali OBB podatkov.")

    return oznake