import json
import asyncio
from typing import Set

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

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