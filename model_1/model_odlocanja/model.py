from ultralytics import YOLO


def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt"):
    """naloži YOLOv8 model iz podane poti ali URL-ja."""
    model = YOLO(url)
    return model

model = nalozi_model()

PARKING_LINE_CLASS_ID = 7

def obdelaj_sliko(frame, sigurnost = 0.6):
    """
    obdelaj posamezen okvir (sliko) in vrni:
    - označen okvir (annotated image)
    - rezultate detekcije
    """
    results = model(frame, verbose=False, conf=sigurnost)

    annotated = results[0].plot()


    return annotated, results[0], izracunaj_ovire(results[0])


def izracunaj_ovire(results) -> 1 | 2 | 3:
    """
    Vrne:
    - 1: nič nevarnosti
    - 2: objekt je v sredinskih 60% (x)
    - 3: objekt je v sredini + spodnjih 20% (y)
    """

    if not hasattr(results, 'obb') or results.obb is None or results.obb.xywhr is None or len(results.obb.xywhr) == 0:
        return 1

    image_width = results.orig_shape[1]
    image_height = results.orig_shape[0]

    left_boundary = image_width * 0.2
    right_boundary = image_width * 0.8
    bottom_threshold = image_height * 0.8

    for i, obb_data_row in enumerate(results.obb.xywhr):
        cls_id = results.obb.cls[i].item()
        if cls_id == PARKING_LINE_CLASS_ID:
            continue

        x_center = obb_data_row[0].item()
        y_center = obb_data_row[1].item()
        width = obb_data_row[2].item()
        height = obb_data_row[3].item()

        # estimate object bounds
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_bottom = y_center + height / 2  # bottom edge of the OBB

        # is in middle 60%?
        if x_min < right_boundary and x_max > left_boundary:
            if y_bottom >= bottom_threshold:
                return 3
            else:
                return 2

    return 1

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