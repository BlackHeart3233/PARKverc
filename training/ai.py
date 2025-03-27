from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # ali 's', 'm', 'l', 'x' za večje modele

model.train(data="data.yaml", epochs=50, imgsz=640)
