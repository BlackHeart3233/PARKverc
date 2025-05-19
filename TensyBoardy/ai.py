from ultralytics import YOLO

# Use detection model (not segmentation)
model = YOLO("yolov8n.pt")  # Or yolov8s.pt, yolov8m.pt etc.

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16  # or adjust based on your GPU
)
