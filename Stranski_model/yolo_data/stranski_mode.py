from ultralytics import YOLO

def nalozi_model(url: str = "https://huggingface.co/ParkVerc/model_stranski/blob/main/weights/best.pt"):
    """naloži YOLOv8 model iz podane poti ali URL-ja."""
    model = YOLO(url)
    return model

model = nalozi_model()

def obdelaj_sliko_model_2(frame, sigurnost = 0.6):
    """
    obdelaj posamezen okvir (sliko) in vrni:
    - označen okvir (annotated image)
    - rezultate detekcije
    """
    results = model(frame, verbose=False, conf=sigurnost)

    annotated = results[0].plot()

    return annotated, results[0]