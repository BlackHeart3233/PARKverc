import cv2
import argparse
from ultralytics import YOLO

from model_odlocanja.model import obdelaj_sliko

"""
Argumenti:

--source webcam
--source path_do_video_datoteke
"""

parser = argparse.ArgumentParser(description="YOLOv8 Live / Video Detection")
parser.add_argument('--source',
                    type=str,
                    default='video1.mov',
                    help="Input source: 'webcam' or path to video file")

args = parser.parse_args()

SKIP_FRAMES = 50

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

frame_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break


    if frame_count % (SKIP_FRAMES + 1) == 0:
        annotated_frame, result = obdelaj_sliko(frame)

    frame_count += 1


    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
