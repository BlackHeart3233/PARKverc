from ultralytics import YOLO
import cv2
from typing import Union
from fastapi import FastAPI, Request, Response,WebSocket
import numpy as np
from PIL import Image
import os
import time
from bitstring import BitArray
import threading
import cv2
import numpy as np
import compressor
import json
import asyncio
from typing import Set
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

latest_frame = None

app = FastAPI()


app.mount(
    "/stranski_public",
    StaticFiles(directory="public", html=True),
    name="stranski_public"
)

app.mount(
    "/vzvratni_public",
    StaticFiles(directory="vzvratni_public", html=True),
    name="vzvratni_public"
)

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
#python -m uvicorn simulacija_obdelava:app --reload --host 0.0.0.0 --port 8000

#VZRATNI MODEL
model = YOLO("https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt")
PARKING_LINE_CLASS_ID = 7

@app.post("/obdelaj_sliko")
async def obdelaj_sliko(request: Request):
    global latest_frame

    data = {"lines": []}

    compressed = await request.body()
    if not compressed:
        return {"error": "empty body"}

    #posiljam barvne slike 
    arr = np.frombuffer(compressed, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)


    if img is None:
        return {"error": "invalid image"}    

    ts = int(time.time() * 1000)
    original_path = os.path.join(DEBUG_DIR, f"{ts}_original.jpg")
    cv2.imwrite(original_path, img)

    """
    try:
        gray = compressor.decompress(compressed)
        gray = np.asarray(gray, dtype=np.uint8)
    except Exception as e:
        print("DECOMPRESS ERROR:", e)
        return {"error": "decompress failed"}

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    """

    result = model(img, verbose=False, conf=0.6)[0]

    if result.obb is not None:
        for i, obb_data_row in enumerate(result.obb.xywhr):
            cls_id = int(result.obb.cls[i].item())
            if cls_id != PARKING_LINE_CLASS_ID:
                continue

            x_center, y_center, width, height, angle = [
                obb_data_row[j].item() for j in range(5)
            ]

            box = (
                (x_center, y_center),
                (width, height),
                angle * 180 / np.pi
            )

            box_pts = cv2.boxPoints(box)
            box_pts = np.intp(box_pts)

            cv2.polylines(
                img,
                [box_pts],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            coords_list = box_pts.tolist()
            data["lines"].append({
                "corners": [
                    {"x": int(x), "y": int(y)} for x, y in coords_list
                ]
            })

    latest_frame = img.copy()

    return {"json": data}


@app.post("/kompresija")
async def kompresija(request: Request):
    raw = await request.body()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Invalid image"}
    compressed = compressor.compress(img)
    return Response(
        content=compressed.tobytes(),
        media_type="application/octet-stream"
    )

   
from fastapi.responses import StreamingResponse

def mjpeg_generator():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        _, jpg = cv2.imencode(".jpg", latest_frame)
        frame = jpg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )
        time.sleep(0.03)  # ~30 FPS


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


#stranski model   - Lovro 



DATA_FILE = "yolo_data.txt"
SEND_INTERVAL_SECONDS = 0.2

clients: Set[WebSocket] = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("Client connected")

    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in clients:
            clients.remove(websocket)


async def broadcast(message: dict):
    dead_clients = []
    for ws in list(clients):
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending to client: {e}")
            dead_clients.append(ws)

    for ws in dead_clients:
        if ws in clients:
            clients.remove(ws)


async def file_loop():
    while True:
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        detections = json.loads(line)
                    except json.JSONDecodeError:
                        detections = line

                    await broadcast({"detections": detections})

                    await asyncio.sleep(SEND_INTERVAL_SECONDS)

            print("eof, restarting")

        except FileNotFoundError:
            print(f"{DATA_FILE} not found")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error in file_loop: {e}")
            await asyncio.sleep(2)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(file_loop())

