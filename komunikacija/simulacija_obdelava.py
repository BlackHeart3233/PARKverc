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
import serial
import base64
import httpx
"""
Pred zagonom preveri PORT in ga spremeni glede na STM32.
Nato zaženi s pomočjo ukaza:
python -m uvicorn simulacija_obdelava:app --reload --host 0.0.0.0 --port 8000

zadnji model: http://localhost:8000/vzvratni_public/
za pravilno delovanje zadnjega zazeni se client.py v katerem preveri folder s frame slikami

stranski model: http://localhost:8000/stranski_public/
za pravilno delovanje strani modela je potrebno zagnati producer.py
in reload site
"""

PORT = "COM6"
BAUD = 115200

ser = None

try:
    ser = serial.Serial(
        port=PORT,
        baudrate=BAUD,
        timeout=1
    )
    print(f"[SERIAL] Connected to {PORT}")
except serial.SerialException as e:
    print(f"[SERIAL] WARNING: {PORT} not available ({e})")
    ser = None


latest_rot = None
latest_dist = None
serial_lock = threading.Lock()


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

#VZRATNI MODEL
model = YOLO("https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt")
PARKING_LINE_CLASS_ID = 7


sensor_socket: WebSocket | None = None
web_socket: WebSocket | None = None


#glavni server

@app.websocket("/handler")
async def handler(socket: WebSocket):
    global sensor_socket, web_socket
    await socket.accept()
    print("Client povezan")

    try:
        while True:
            msg = await socket.receive()

            if msg.get("text") is not None:
                text = msg["text"]
            elif msg.get("bytes") is not None:
                text = msg["bytes"].decode("utf-8")
            else:
                continue

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                if text == "WEB":
                    web_socket = socket
                    print("WEB povezan")
                    continue

                if text == "SENZOR":
                    sensor_socket = socket
                    print("SENZOR povezan")
                    continue

                print("Neznan tekst:", text)
                continue

            if obj.get("type") == "KAMERA":
                if not obj.get("data"):
                    print("KAMERA brez data")
                    continue

                compressed_bytes = base64.b64decode(obj["data"])
                print("Prejet KAMERA frame:", len(compressed_bytes), "bytes")
                result = obdelaj_sliko(compressed_bytes)

                if web_socket:
                    await web_socket.send_text(json.dumps({
                        "type": "FRAME",
                        "json": result["json"],
                        "image": result["image"]
                    }))
                else:
                    print("WEB ni povezan")


    except WebSocketDisconnect:
        print("Client odklopljen")

    finally:
        if socket == web_socket:
            web_socket = None
            print("WEB odklopljen")

        if socket == sensor_socket:
            sensor_socket = None
            print("SENZOR odklopljen")


last_sent = (None, None)

async def send_sensor_loop():
    global last_sent
    while True:
        await asyncio.sleep(0.1)

        if web_socket is None:
            continue

        with serial_lock:
            rot = latest_rot
            dist = latest_dist

        if rot is None or dist is None:
            continue

        if (rot, dist) == last_sent:
            continue

        last_sent = (rot, dist)
        try:
            await web_socket.send_text(json.dumps({
                "type": "SENZOR_DATA",
                "Rotary": round(rot, 2),
                "Distance": round(dist, 2)
            }))
        except Exception:
            pass



#zagon posiljanja senzorja
@app.on_event("startup")
async def startup():
    asyncio.create_task(send_sensor_loop())







def obdelaj_sliko(compressed: bytes) -> dict:
    data = {"lines": []}

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

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buffer = cv2.imencode(".jpg", img, encode_param)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "json": data,
        "image": img_b64   
    }



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
    print("rpi connected")

    try:
        while True:
            compressed: bytes = await ws.receive_bytes()

            try:
                gray = compressor.decompress(compressed)
                gray = np.asarray(gray, dtype=np.uint8)
            except Exception as e:
                print("decompression failed:", e)
                continue

            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            result = stranski_model(frame)[0]
            annotated = result.plot()

            h, w = frame.shape[:2]
            detections = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < 0.25:
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # izboljšave
                height_ratio = (y2 - y1) / h
                bottom_ratio = y2 / h

                if height_ratio < 0.15:
                    continue

                if height_ratio < 0.25:
                    row = "back"
                else:
                    if bottom_ratio >= 0.65:
                        row = "front"
                    else:
                        row = "back"

                detections.append({
                    "label": stranski_model.names[cls],
                    "confidence": conf,
                    "left_to_right": max(0, min(100, (cx / w) * 100)),
                    "up_to_down": max(0, min(100, (cy / h) * 100)),
                    "down_to_up": 100 - max(0, min(100, (cy / h) * 100)),
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "size": {"width": x2 - x1, "height": y2 - y1},
                    "depth_row": row
                })

            _, buf = cv2.imencode(".jpg", cv2.resize(annotated, (350, 230)))
            image_b64 = base64.b64encode(buf).decode()

            await broadcast({
                "detections": detections,
                "image": image_b64
            })

    except WebSocketDisconnect:
        print("rpi disconnected")

#STM32

@app.websocket("/ws/rotate")
async def ws_rotate(ws: WebSocket):
    await ws.accept()
    print("Rotate WS connected")

    try:
        while True:
            msg = await ws.receive_text()

            try:
                deg = float(msg)
            except ValueError:
                await ws.send_text("ERROR: invalid number")
                continue

            if ser is None:
                await ws.send_text("ERROR: Serial not available")
                continue

            cmd = f"ROTATE:{deg}\n"
            ser.write(cmd.encode("ascii"))

            await ws.send_text(f"OK: sent {deg}")

    except WebSocketDisconnect:
        print("Rotate WS disconnected")

def rx_thread():
    global latest_rot, latest_dist
    
    if ser is None:
        print("[SERIAL] RX thread disabled (no serial)")
        return

    while True:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue

            # Example: "ROT:12.3 DIST:45.6"
            parts = line.split()

            rot = None
            dist = None

            for p in parts:
                if p.startswith("ROT:"):
                    rot = float(p.split(":")[1])
                elif p.startswith("DIST:"):
                    dist = float(p.split(":")[1])

            if rot is not None and dist is not None:
                print(f"  -> Rotary = {rot:.2f} deg | Distance = {dist:.2f} cm")

            with serial_lock:
                if rot is not None:
                    latest_rot = rot
                if dist is not None:
                    latest_dist = dist

        except Exception as e:
            print("Serial RX error:", e)
            break

if ser is not None:
    threading.Thread(target=rx_thread, daemon=True).start()
else:
    print("[SERIAL] RX thread not started (no serial)")


def iou(boxA, boxB):
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    boxBArea = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

    return interArea / float(boxAArea + boxBArea - interArea)

def suppress_duplicates(detections, iou_thresh=0.6):
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in detections:
        duplicate = False
        for k in kept:
            if det["label"] == k["label"]:
                if iou(det["coordinates"], k["coordinates"]) > iou_thresh:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)

    return kept
