from ultralytics import YOLO
import cv2
import numpy as np
#import paho.mqtt.publish as publish
'''
def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_stranski/blob/main/weights/best.pt"):
    """naloži YOLOv8 model iz podane poti ali URL-ja."""
    model = YOLO(url)
    return model
'''
def nalozi_model():
    model = YOLO(r"../assets/Kjara/best_test.pt")
    return model

from ultralytics import YOLO
import cv2
import os

BROKER = "10.0.0.1"
TOPIC1 = "spo/stevilo_label_na_frame"
TOPIC2 = "spo/prosto_parkirno_mesto"
TOPIC3 = "spo/zasedeno_parkirno_mesto"

can_park = cv2.resize(
    cv2.imread("../assets/Kjara/signs/can_park.jpg", cv2.IMREAD_UNCHANGED),
    (125, 125),  #širina, višina v pikslih 
    interpolation=cv2.INTER_AREA
)
family_car = cv2.resize(
    cv2.imread("../assets/Kjara/signs/family_car.jpg", cv2.IMREAD_UNCHANGED),
    (125, 125),  #širina, višina v pikslih 
    interpolation=cv2.INTER_AREA
)
electric_car = cv2.resize(
    cv2.imread("../assets/Kjara/signs/electric_car.jpg", cv2.IMREAD_UNCHANGED),
    (125, 125),  #širina, višina v pikslih
    interpolation=cv2.INTER_AREA
)
car= cv2.resize(
    cv2.imread("../assets/Kjara/signs/car.webp", cv2.IMREAD_UNCHANGED),
    (80, 80),  #širina, višina v pikslih
    interpolation=cv2.INTER_AREA
)
handicap_parking= cv2.resize(
    cv2.imread("../assets/Kjara/signs/car.webp", cv2.IMREAD_UNCHANGED),
    (80, 80),  #širina, višina v pikslih
    interpolation=cv2.INTER_AREA
)
def izpisi_obb_info(result, model):
    obb = result.obb
    if obb is not None:
        # Pretvori cls in xywhr v numpy
        cls_ids = obb.cls.cpu().numpy() if obb.cls is not None else []
        coords = obb.xywhr.cpu().numpy() if hasattr(obb, 'xywhr') else None
        confs = obb.conf.cpu().numpy() if obb.conf is not None else []

        if len(cls_ids) > 0 and coords is not None:
            print(f"🔢 Zaznanih OBB objektov: {len(cls_ids)}")
            for i, cls_id in enumerate(cls_ids):
                label_name = model.names[int(cls_id)]
                coord = coords[i]
                conf = confs[i] if i < len(confs) else None
                print(f"✅ Razred: {label_name}, Koordinate OBB (x_c, y_c, w, h, rot): {coord}, Confidence: {conf}")
        else:
            print("⚠️ OBB atributi so prazni ali niso dostopni.")
    else:
        print("⚠️ OBB ni zaznan.")


model = nalozi_model()

def obdelaj_sliko_model_2_1(frame, sigurnost = 0.6):
    """
    obdelaj posamezen okvir (sliko) in vrni:
    - označen okvir (annotated image)
    - rezultate detekcije
    """
    results = model(frame, verbose=False, conf=sigurnost)

    annotated = results[0].plot()

    return annotated, results[0]

def obdelaj_sliko_model_2(frame, sigurnost=0.6):
    results = model(frame, verbose=False, conf=sigurnost)
    result = results[0]
    annotated = frame.copy()

    labels = []  # tukaj bomo zbirali vse labele

    boxes = result.boxes
    steviloLabelov = len(boxes) if boxes is not None else 0
    free_parking = 0
    occupied_parking = 0

    if boxes is not None and boxes.xyxy is not None:
        coords = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i, (x1, y1, x2, y2) in enumerate(coords):
            cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
            label = model.names.get(cls_id, "Neznano")
            conf = confs[i]

            labels.append(label)  # shrani label v seznam

            ikona = None
            if label == "Prosto_parkirno_mesto":
                ikona = can_park
                free_parking+=1
            elif label == "Družinsko_parkiranje":
                ikona = family_car
                occupied_parking+=1
            elif label == "Električno_parkiranje":
                ikona = electric_car
                occupied_parking+=1
            elif label == "Invalidsko_parkiranje":
                ikona = handicap_parking
                occupied_parking+=1
            elif label == "Avtomobil":
                ikona = car
                occupied_parking+=1


            if ikona is not None:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                top_left_x = center_x - ikona.shape[1] // 2
                top_left_y = center_y - ikona.shape[0] // 2

                if (0 <= top_left_x < frame.shape[1] - ikona.shape[1]) and (0 <= top_left_y < frame.shape[0] - ikona.shape[0]):
                    overlay = annotated[top_left_y:top_left_y + ikona.shape[0], top_left_x:top_left_x + ikona.shape[1]]

                    if ikona.shape[2] == 4:
                        alpha = ikona[:, :, 3] / 255.0
                        for c in range(3):
                            overlay[:, :, c] = (1 - alpha) * overlay[:, :, c] + alpha * ikona[:, :, c]
                    else:
                        overlay[:, :, :] = ikona

                    annotated[top_left_y:top_left_y + ikona.shape[0], top_left_x:top_left_x + ikona.shape[1]] = overlay
    else:
        print("⚠️ Ni zaznanih 'boxes' objektov.")

    #publish.single(TOPIC1, str(steviloLabelov), hostname=BROKER)
    #publish.single(TOPIC2, str(free_parking), hostname=BROKER)
    #publish.single(TOPIC3, str(occupied_parking),hostname=BROKER)


    return annotated, labels  # vrni seznam vseh labelov


# 🔽 TESTNA FUNKCIJA
if __name__ == "__main__":
    pot_do_slike = "../assets/Kjara/images_from_video/Video_009_25_4_2025/frame_155.jpg"

    if not os.path.exists(pot_do_slike):
        print(f"❌ Napaka: Pot do slike ne obstaja -> {pot_do_slike}")
    else:
        frame = cv2.imread(pot_do_slike)

        if frame is None:
            print("❌ Napaka: Slika ni bila uspešno naložena. Preveri format ali pot.")
        else:
            označena, rezultat = obdelaj_sliko_model_2(frame, sigurnost=0.6)
            cv2.imshow("🔍 Detekcija", označena)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

