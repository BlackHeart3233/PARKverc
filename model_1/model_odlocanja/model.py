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
    Preveri, ali je katerokoli detektirano polje (box) v srednjih 40% širine zaslona,
    using OBB (Oriented Bounding Box) data.

    Args:
        results: Rezultati detekcije iz YOLO modela (results[0] iz obdelaj_sliko).

    Returns:
        True, če je vsaj eno polje v srednjih 40% širine zaslona, sicer False.
    """

    # check if 'obb' attribute exists and contains detections
    # results.obb will be an OBB object, not directly a list
    # t has properties like xywhr, conf, cls etc.
    # we need to check if obb.xywhr is not None and has length.
    if not hasattr(results, 'obb') or results.obb is None or results.obb.xywhr is None or len(results.obb.xywhr) == 0:
        return False

    image_width = results.orig_shape[1]  # širina originalne slike
    image_height = results.orig_shape[0]  # višina originalne slike

    # določitev območja srednjih 40% -60%, PO DOMAČE JE NA SREDINI
    left_boundary = image_width * 0.4
    right_boundary = image_width * 0.6

    # določitev zgornje meje
    top_boundary = image_height * 0.9

    danger_level = 0

    # rows of the xywhr tensor
    # each row represents [x_center, y_center, width, height, angle] for one OBB
    for i, obb_data_row in enumerate(results.obb.xywhr):

        current_class_id = results.obb.cls[i].item()

        if current_class_id == PARKING_LINE_CLASS_ID:
            continue

        x_center = obb_data_row[0].item()  # first element of the row, then .item()
        y_center = obb_data_row[1].item()  # first element of the row, then .item()

        print(x_center, y_center)

        # preverimo, ali je center_x znotraj mej
        if right_boundary >= x_center >= left_boundary:
            if danger_level < 2:
                danger_level = 2
            else:
                if y_center <= top_boundary:
                    danger_level = 3


    #print(danger_level)
    return danger_level

