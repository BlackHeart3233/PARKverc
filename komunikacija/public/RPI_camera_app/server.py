import asyncio
import websockets
import numpy as np
import cv2
import compressor

viewers = set()

async def camera_handler(ws):
    print("📷 Camera connected")
    try:
        async for msg in ws:
            # msg is compressed bytes
            compressed = msg

            # ---- decompress ----
            img = compressor.decompress(compressed)
            # img: numpy array (H, W), uint8

            # ---- encode to PNG ----
            ok, png = cv2.imencode(".png", img)
            if not ok:
                continue

            data = png.tobytes()

            # ---- send to all viewers ----
            for v in viewers.copy():
                try:
                    await v.send(data)
                except:
                    viewers.remove(v)

    except websockets.ConnectionClosed:
        print("❌ Camera disconnected")

async def viewer_handler(ws):
    print("👁 Viewer connected")
    viewers.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        viewers.remove(ws)
        print("👁 Viewer disconnected")

async def main():
    async with websockets.serve(camera_handler, "0.0.0.0", 8000, path="/ws/camera"):
        async with websockets.serve(viewer_handler, "0.0.0.0", 8000, path="/ws/viewer"):
            print("🚀 Server running")
            await asyncio.Future()

asyncio.run(main())
