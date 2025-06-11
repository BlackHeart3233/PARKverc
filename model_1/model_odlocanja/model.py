from ultralytics import YOLO
import cv2
import numpy as np

def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt"):
    model = YOLO(url)
    return model

model = nalozi_model()

def obdelaj_sliko(frame, sigurnost=0.6):
    results = model(frame, verbose=False, conf=sigurnost)
    result = results[0]
    annotated = result.plot()
    return annotated, result

def izpisi_in_izlusci(result):
    oznake = []

    obb = result.obb
    if obb is not None and obb.cls is not None and obb.xywhr is not None:
        cls_ids = obb.cls.cpu().numpy()
        coords = obb.xywhr.cpu().numpy()
        confs = obb.conf.cpu().numpy() if obb.conf is not None else [-1.0]*len(cls_ids)

        print(f"🔷 Detektiranih OBB objektov: {len(cls_ids)}")
        for i, cls_id in enumerate(cls_ids):
            label = model.names[int(cls_id)]
            coord = coords[i]
            conf = confs[i] if i < len(confs) else -1
            print(f"OBB: ✅ Razred: {label}, Koordinate: {coord}, Confidence: {conf:.2f}")

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

        print(f"🔳 Detektiranih standardnih boxov: {len(class_ids)}")
        for i, cls_id in enumerate(class_ids):
            label = model.names[int(cls_id)]
            coord = coords[i]
            conf = confs[i] if i < len(confs) else -1
            print(f"Box: ✅ Razred: {label}, Koordinate: {coord}, Confidence: {conf:.2f}")

            oznake.append({
                "type": "box",
                "label": label,
                "class_id": int(cls_id),
                "confidence": float(conf),
                "bbox": coord.tolist()
            })

    else:
        print("⚠️ Ni zaznanih oznak ali OBB podatkov.")

    return oznake
