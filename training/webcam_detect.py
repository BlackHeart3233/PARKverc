import cv2
import threading
from ultralytics import YOLO
import time

model = YOLO("runs/detect/train3/weights/last.pt")  #  speed
#model = YOLO("runs/detect/train3/weights/best.pt")  # accuracy

# frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# init stream
stream = WebcamStream().start()
time.sleep(1.0)  # camera should warm up

print("Live Detection started ctrl + c to stop")

while True:
    frame = stream.read()
    if frame is None:
        continue

    # opcijsko: resize for speed (model itak auto-resize-a)
    frame_resized = cv2.resize(frame, (640, 480))

    # disable tracking for more speed
    results = model(frame_resized, verbose=False)

    # nraiši rezultat
    annotated = results[0].plot()

    # pokaž v okno
    cv2.imshow("YOLOv8 livestream", annotated)

    # izhod
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
