from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")
results = model("test/frame_29.jpg", save=True, conf=0.05)
