import cv2
import argparse
from ultralytics import YOLO


"""
Argumenti:

--source webcam
--source path_do_video_datoteke
"""

parser = argparse.ArgumentParser(description="YOLOv8 Live / Video Detection")
parser.add_argument('--source',
                    type=str,
                    default='webcam',
                    help="Input source: 'webcam' or path to video file")

args = parser.parse_args()


model = YOLO("runs/detect/train/weights/best.pt")


if args.source == 'webcam':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("webcam input.")
else:
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Could not open video file: {args.source}")
        exit()
    print(f"Using video file: {args.source}")

print("ctrl + c  to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
