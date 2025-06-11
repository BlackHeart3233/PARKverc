from ultralytics import YOLO

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

model = nalozi_model()

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

    if debugger_mode:
        return frame, results[0], izracunaj_ovire(results[0])
    else:
        return annotated, results[0], izracunaj_ovire(results[0])


def izracunaj_ovire(results) -> int:
    if not hasattr(results, 'obb') or results.obb is None or results.obb.xywhr is None or len(results.obb.xywhr) == 0:
        return 0

    image_width = results.orig_shape[1]
    image_height = results.orig_shape[0]

    left_boundary = image_width * 0.3
    right_boundary = image_width * 0.7
    bottom_threshold = image_height * 0.7

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
