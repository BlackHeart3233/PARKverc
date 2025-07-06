from ultralytics import YOLO
import cv2
import numpy as np

def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt"):
    """naloži YOLOv8 model iz podane poti ali URL-ja."""
    model = YOLO(url)
    return model

model = nalozi_model()

PARKING_LINE_CLASS_ID = 7
ZEBRA_CLASS_ID = 11

def obdelaj_sliko(frame, sigurnost = 0.6):
    """
    obdelaj posamezen okvir (sliko) in vrni:
    - označen okvir (annotated image)
    - rezultate detekcije
    """
    results = model(frame, verbose=False, conf=sigurnost)

    annotated = results[0].plot()


    return annotated, results[0], izracunaj_ovire(results[0]), narisi_parking_boxes(results[0], frame)


def izracunaj_ovire(results) -> int:
    """
    Vrne:
    - 1: nič nevarnosti
    - 2: objekt v sredinskih 50% (x)
    - 3: spodnji rob objekta je v sredini (x) + spodnjih 10% (y)
    """
    if not hasattr(results, 'obb') or results.obb is None or results.obb.xywhr is None or len(results.obb.xywhr) == 0:
        return 1

    image_width = results.orig_shape[1]
    image_height = results.orig_shape[0]

    center_left = image_width * 0.2
    center_right = image_width * 0.8
    bottom_threshold = image_height * 0.9

    highest_danger = 0

    for i, obb_data_row in enumerate(results.obb.xywhr):
        cls_id = results.obb.cls[i].item()
        if cls_id in [PARKING_LINE_CLASS_ID, ZEBRA_CLASS_ID]:
            continue

        x_center = obb_data_row[0].item()
        y_center = obb_data_row[1].item()
        width = obb_data_row[2].item()
        height = obb_data_row[3].item()

        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_bottom = y_center + height / 2

        if x_min < center_right and x_max > center_left:

            if x_min < image_width * 0.7 and x_max > (image_width * 0.3):
                if y_bottom >= bottom_threshold:
                    if highest_danger < 3:
                        highest_danger = 3

            if y_bottom >= image_height * 0.5:
                if highest_danger < 2:
                    highest_danger = 2


            if y_bottom >= image_height * 0.7:
                if highest_danger < 1:
                    highest_danger = 1

    return highest_danger

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

def narisi_parking_boxes(result, image):
    """
    Nariše zelene rotirane okvirje okoli parkirnih črt (class_id == 7).

    Args:
        result: YOLOv8 rezultat (results[0])
        image: originalna slika (cv2 image)

    Returns:
        image z narisanimi rotiranimi zelenimi okvirji
    """
    if not hasattr(result, 'obb') or result.obb is None or result.obb.xywhr is None:
        return image

    for i, obb_data_row in enumerate(result.obb.xywhr):
        cls_id = result.obb.cls[i].item()
        if cls_id != 7:
            continue  # samo za parkirne črte

        x_center = obb_data_row[0].item()
        y_center = obb_data_row[1].item()
        width = obb_data_row[2].item()
        height = obb_data_row[3].item()
        angle = obb_data_row[4].item()

        box = ((x_center, y_center), (width, height), angle * 180 / np.pi)
        box_pts = cv2.boxPoints(box)
        box_pts = np.intp(box_pts)

        cv2.drawContours(image, [box_pts], 0, (0, 255, 0), 2)

    return image
