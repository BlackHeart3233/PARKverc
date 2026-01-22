import cv2
import asyncio
import websockets
import compressor

WS_URL = "ws://localhost:8000/ws/producer"
VIDEO_PATH = "video2.mp4"
FPS = 4
DELAY = 1.0 / FPS


async def send_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Ni uspelo odpreti videa"

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("Connected to server")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        skip = max(1, int(video_fps / FPS))

        while True:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # crop bottom 20%
            valid_h = int(gray.shape[0] * 0.8)
            gray = gray[:valid_h, :]

            original_bytes = gray.nbytes

            # compress
            compressed = compressor.compress(gray)
            compressed_bytes = compressed.tobytes()
            compressed_size = len(compressed_bytes)

            reduction_pct = (1 - compressed_size / original_bytes) * 100

            # send compressed bytes
            await ws.send(compressed_bytes)

            print(
                f"➡️ poslan frame {gray.shape} | "
                f"raw={original_bytes}B → comp={compressed_size}B | "
                f"zmanjšanje={reduction_pct:.1f}%"
            )

            await asyncio.sleep(DELAY)

            # skip frames to keep target FPS
            for _ in range(skip - 1):
                cap.read()


asyncio.run(send_video())
