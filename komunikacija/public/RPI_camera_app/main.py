import asyncio
import websockets
from picamera2 import Picamera2

from compression import stisni_sliko_bytes  # <-- HERE

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

async def main():
    uri = "ws://PC_IP:8000/ws/camera"
    async with websockets.connect(uri, max_size=None) as ws:
        while True:
            frame = picam2.capture_array()          # RGB array from picamera2
            data = stisni_sliko_bytes(frame, 0)     # <-- compress to bytes
            await ws.send(data)
            await asyncio.sleep(0.5)

asyncio.run(main())
