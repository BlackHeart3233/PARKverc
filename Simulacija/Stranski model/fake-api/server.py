"""
navodila za pogAnanje:
1. venv
- python -m venv venv
- venv\Scripts\activate

2. namestitev paketov
pip install fastapi uvicorn ultralytics opencv-python

pip install -r requirements.txt

3. zagon serverja
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

4. povezava iz browserja ali aplikacije na ws://localhost:8000/ws


5. naredil sem yolo_data.txt da ne rabimo vedno poganjat modela -> lahko prebereš iz datoteke in simuliraš websocket sporočila
- samo odkomentiraj ustrezne dele v video_loop funkciji če želiš dejansko funkcionalnost

6. imamo main.py server za smo predvajanje / testiranje iz tesntih podtkov
"""


import cv2
import json
import asyncio
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import threading
import time
import json

app = FastAPI()

# stranski model
model = YOLO("https://huggingface.co/ParkVerc/model_stranski/resolve/main/stranski_model_augmentiran/weights/last.pt")  # <-- replace with local path for speed

clients = set()

DATA_FILE = "yolo_data.txt"

def save_line(filepath, data):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("Client connected")

    try:
        while True:
            await asyncio.sleep(1)
    except:
        clients.remove(websocket)
        print("Client disconnected")


async def broadcast(message):
    dead = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(message))
        except:
            dead.append(ws)

    for d in dead:
        clients.remove(d)


def video_loop():
    cap = cv2.VideoCapture("video2.mp4")

    if not cap.isOpened():
        print("wrong video")
        return

    SKIP = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # yolo detekcija
        result = model(frame)[0]

        # tole odkomentiraj za real time prikaz
        annotated = result.plot()
        preview = cv2.resize(annotated, (600, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow("Processing Preview (600x400)", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        h, w = frame.shape[:2]

        ## projection matrix picking
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # CENTER X
            center_x = (x1 + x2) / 2

            # NORMALIZED LEFT->RIGHT POSITION (0 = lev, 100 = desno)
            horizontal_pos = (center_x / w) * 100
            horizontal_pos = max(0, min(100, horizontal_pos))

            vertical_pos = (center_x / w) * 100
            vertical_pos = max(0, min(100, vertical_pos))


            coordinates = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }

            size = {
                "width": x2 - x1,
                "height": y2 - y1
            }

            detections.append({
                "label": model.names[cls],
                "confidence": conf,
                "left_to_right": horizontal_pos,
                "up_to_down": vertical_pos,
                "down_to_up": 100 - vertical_pos,
                "coordinates": coordinates,
                "size": size
            })

        # websocket broadcast
        asyncio.run(broadcast({"detections": detections}))

        save_line(DATA_FILE, detections)

        # preskočimo par frejmov da si prišparamo malo cpu časa
        current = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current + SKIP)

        # okoli 1 na sekund
        time.sleep(0.2)

# v odzadju procesiramo
threading.Thread(target=video_loop, daemon=True).start()