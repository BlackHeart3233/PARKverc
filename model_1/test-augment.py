import cv2
import os
import glob
import numpy as np

img_dir = "augmented/images"
label_dir = "augmented/labels"

image_files = glob.glob(f"{img_dir}/*.jpg")

for img_path in image_files:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, f"{base}.txt")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls, coords = parts[0], list(map(float, parts[1:]))
            pts = np.array([[int(x * w), int(y * h)] for x, y in zip(coords[::2], coords[1::2])], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Augmented OBB Preview", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
