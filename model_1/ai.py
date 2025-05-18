from ultralytics import YOLO

def train():
    model = YOLO("yolov8s-obb.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        amp=True
    )

if __name__ == "__main__":
    train()
