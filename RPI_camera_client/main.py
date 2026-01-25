import asyncio
import websockets
import cv2
import numpy as np
import compressor

VIDEO_PATH = "video.MP4"
TARGET_FPS = 5
FRAME_DELAY = 1.0 / TARGET_FPS


async def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Ni uspelo odpreti videa"

    uri = "ws://192.168.50.243:8000/ws/camera"
    async with websockets.connect(uri, max_size=None) as ws:
        print("povezan na ws")

        while True:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            valid_h = int(gray.shape[0] * 0.8)
            gray = gray[:valid_h, :]

            original_bytes = gray.nbytes

            compressed = compressor.compress(gray)
            compressed_bytes = compressed.tobytes()
            compressed_size = len(compressed_bytes)

            reduction_pct = (1 - compressed_size / original_bytes) * 100

            await ws.send(compressed_bytes)

            print(
                f"➡️ poslan frame {gray.shape} | "
                f"raw={original_bytes}B → comp={compressed_size}B | "
                f"zmanjšanje={reduction_pct:.1f}%"
            )

            await asyncio.sleep(FRAME_DELAY)


asyncio.run(main())
