from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="Stranski_model/yolo_data_2/split_train_2/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="stranski_model_earlystop",
    early_stopping=5  #overfitting validacijske za val/loss ce se ne izboljsa v 5
)





'''from ultralytics import YOLO #tole samo testira model

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on a folder or single image
results = model.predict(source="assets/Kjara/images_from_video/Video_007_25_4_2025/frame_25.jpg", save=True, conf=0.25)

# Show results
for r in results:
    r.show()  # open image with boxes'''