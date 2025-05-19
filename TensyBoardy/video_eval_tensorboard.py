import cv2
import os
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# 🔧 Nastavitve
image_dir = r"C:\Users\nejla\Desktop\TensyBoardy\test_slikice"#pove kje so slike
model_path = r"C:\Users\nejla\Desktop\PARKverc\training\runs\detect\train3\weights\best.pt" #kjw je YOLO model
log_dir = "runs/image_eval" #kam naj shrani slike

class_names = {
    0: "Ball", 1: "Bicycle", 2: "Bike", 3: "Car", 4: "Cart",
    5: "Curb", 6: "Human", 7: "Parking_line", 8: "Pole", 9: "Ramp", 10: "Tree"
}

model = YOLO(model_path) #tu povežeš modell
writer = SummaryWriter(log_dir=log_dir) #odpreš kanal za pisanje podatkov v tensor board

# pregleda celotno podano mapo
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # da ohraniš urejenost

print("Začenjam z analizo slik...")

for idx, filename in enumerate(image_files):  #gre slikico po slikico
    img_path = os.path.join(image_dir, filename)
    image = cv2.imread(img_path) #prebere sliko

    if image is None:
        print(f"Ne morem naložiti slike: {img_path}")
        continue

    results = model(image, verbose=False)[0] #yolo model poišče objekte na sliki in jih razvrsti v razrede

    class_counts = defaultdict(int) #prešteje št objektov na sliki
    for det in results.boxes.data.tolist():
        cls_id = int(det[5])
        class_counts[cls_id] += 1

    for cls_id, count in class_counts.items():
        class_name = class_names.get(cls_id, f"class_{cls_id}")
        writer.add_scalar(f"Detections/{class_name}", count, idx) #zapiše kar je blo detektirano v tensor board
        #vsak razred dobi scoj stolpec s št detekcij

writer.close()
print(f"aključeno. Rezultati zapisani v TensorBoard ({log_dir})")
fdsfsdfsd
