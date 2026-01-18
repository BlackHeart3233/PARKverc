import asyncio
import websockets
import numpy as np
import cv2
import compressor

viewers = set()

import os
import itertools

frame_id = itertools.count()

async def handler(ws):
    path = ws.request.path

    if path == "/ws/camera":
        print("📷 Camera connected")
        try:
            async for compressed in ws:
                img = compressor.decompress(compressed)  # uint8 (H, W)

                ok, png = cv2.imencode(".png", img)
                if not ok:
                    continue

                idx = next(frame_id)
                filename = f"frames/frame_{idx:06d}.png"

                os.makedirs("frames", exist_ok=True)
                with open(filename, "wb") as f:
                    f.write(png.tobytes())

                print(f"💾 saved {filename}")

        except websockets.ConnectionClosed:
            print("📷 Camera disconnected")

    else:
        await ws.close()


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8000):
        print("🚀 Server running")
        await asyncio.Future()

asyncio.run(main())
