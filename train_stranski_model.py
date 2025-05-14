from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
model.train(
    data="assets/Kjara/yolo_data/data.yaml",
    epochs=50,
    imgsz=640
)

#shrani v runs/detect/train/weights/best.pt

'''from ultralytics import YOLO #tole samo testira model

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on a folder or single image
results = model.predict(source="assets/Kjara/images_from_video/Video_007_25_4_2025/frame_25.jpg", save=True, conf=0.25)

# Show results
for r in results:
    r.show()  # open image with boxes'''