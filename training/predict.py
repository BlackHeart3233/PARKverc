from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")
results = model("test/frame_29.jpg", save=True, conf=0.3)
