import cv2
import base64
import time
import asyncio
import websockets

WS_URL = "ws://localhost:8000/ws/producer"
VIDEO_PATH = "video2.mp4"
FPS = 4
DELAY = 1 / FPS


async def send_video():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        cap = cv2.VideoCapture(VIDEO_PATH)
        print("Connected to server")

        if not cap.isOpened():
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        skip = max(1, int(video_fps / FPS))

        while True:
            print("Sending frame")
            ret, frame = cap.read()
            if not ret:
                break

            _, buf = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            await ws.send(img_b64)
            await asyncio.sleep(DELAY)

            # skip frames to keep ~4 FPS
            for _ in range(skip - 1):
                cap.read()

        cap.release()


asyncio.run(send_video())
