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
import base64

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
stranski_model = YOLO("https://huggingface.co/ParkVerc/model_stranski/resolve/main/stranski_model_augmentiran/weights/last.pt")  # <-- replace with local path for speed

frontend_clients: set[WebSocket] = set()

@app.websocket("/ws/frontend")
async def ws_frontend(ws: WebSocket):
    await ws.accept()
    frontend_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    finally:
        frontend_clients.remove(ws)


async def broadcast(payload: dict):
    dead = []
    for ws in frontend_clients:
        try:
            await ws.send_text(json.dumps(payload))
        except:
            dead.append(ws)
    for ws in dead:
        frontend_clients.remove(ws)


@app.websocket("/ws/producer")
async def ws_producer(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            msg = await ws.receive_text()
            img_bytes = base64.b64decode(msg)
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = stranski_model(frame)[0]
            annotated = result.plot()

            h, w = frame.shape[:2]
            detections = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append({
                    "label": stranski_model.names[cls],
                    "confidence": conf,
                    "left_to_right": max(0, min(100, (cx / w) * 100)),
                    "up_to_down": max(0, min(100, (cy / h) * 100)),
                    "down_to_up": 100 - max(0, min(100, (cy / h) * 100)),
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "size": {"width": x2 - x1, "height": y2 - y1},
                })

            _, buf = cv2.imencode(".jpg", cv2.resize(annotated, (350, 230)))
            image_b64 = base64.b64encode(buf.tobytes()).decode()

            await broadcast({
                "detections": detections,
                "image": image_b64
            })

    except Exception:
        pass
