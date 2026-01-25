import asyncio
import websockets
import cv2
import json
import base64
import time

VIDEO_PATH = r"IMG_4905.MP4"
DELAY_MS = 200
WS_URL = "ws://localhost:8000/handler"


async def sensor_client():
    async with websockets.connect(WS_URL) as ws:
        print("Povezan na WS server")

        # identifikacija
        await ws.send("SENZOR")

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("Ne morem odpret videa")
            return

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # JPEG encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ok, buffer = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                continue

            base64_data = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "type": "KAMERA",
                "data": base64_data,
                "frame": frame_idx
            }

            await ws.send(json.dumps(payload))
            print(f"Poslan frame {frame_idx}")

            await asyncio.sleep(DELAY_MS / 1000)

        cap.release()
        print("Video poslan")

        # poslušanje odgovorov (opcijsko)
        try:
            async for msg in ws:
                print("Odgovor strežnika:", msg)
        except websockets.ConnectionClosed:
            print("WS povezava zaprta")


if __name__ == "__main__":
    asyncio.run(sensor_client())
