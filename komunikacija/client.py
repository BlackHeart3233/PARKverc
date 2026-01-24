import asyncio
import websockets
import os
import json
import base64
import re

IMAGE_DIR = r"C:/Users/konjc/OneDrive - Univerza v Mariboru/PARKverc/MERITVE/Meritve_2/Video/Frames_video/IMG_4904"
DELAY_MS = 200  # ms
WS_URL = "ws://localhost:8000/handler"


def extract_number(filename: str) -> int:
    """Izlušči prvo številko iz imena datoteke (za sortiranje)"""
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else 0


async def sensor_client():
    async with websockets.connect(WS_URL) as ws:
        print("Povezan na WS server")

        # identifikacija
        await ws.send("SENZOR")

        files = os.listdir(IMAGE_DIR)
        files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        files.sort(key=extract_number)

        print("Najdenih slik:", len(files))

        for file in files:
            file_path = os.path.join(IMAGE_DIR, file)
            print("Pošiljam:", file)

            with open(file_path, "rb") as f:
                image_bytes = f.read()

            base64_data = base64.b64encode(image_bytes).decode("utf-8")

            payload = {
                "type": "KAMERA",
                "data": base64_data,
                "filename": file
            }

            await ws.send(json.dumps(payload))

            await asyncio.sleep(DELAY_MS / 1000)

        print("Vse slike poslane")

        # če hočeš še poslušat odgovore
        try:
            async for msg in ws:
                print("Odgovor strežnika:", msg)
        except websockets.ConnectionClosed:
            print("WS povezava zaprta")


if __name__ == "__main__":
    asyncio.run(sensor_client())
