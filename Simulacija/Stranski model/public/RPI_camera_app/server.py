import asyncio
import websockets
import cv2
import compressor

async def handler(ws):
    if ws.request.path != "/ws/camera":
        await ws.close()
        return

    print("Camera connected")

    try:
        async for compressed in ws:
            img = compressor.decompress(compressed)
            cv2.imshow("Posnetek", img)
            cv2.waitKey(1)

    except websockets.ConnectionClosed:
        print("Camera disconnected")
        cv2.destroyAllWindows()

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8000):
        print("running")
        await asyncio.Future()

asyncio.run(main())
