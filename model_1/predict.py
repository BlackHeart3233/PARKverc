from ultralytics import YOLO
from model_odlocanja.model import obdelaj_sliko
import cv2

slika = "test/test2.jpg"

annotated_frame, result, _ = obdelaj_sliko(slika)


while True:

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()


cv2.imshow("YOLOv8 Detection", annotated_frame)
