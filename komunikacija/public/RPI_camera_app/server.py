from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

latest_image = None
viewer_connections = set()

@app.get("/")
async def index():
    return HTMLResponse(open("static/viewer.html").read())

@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket):
    global latest_image
    await websocket.accept()
    print("📷 Camera connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            latest_image = data

            # Broadcast to viewers
            for viewer in viewer_connections.copy():
                try:
                    await viewer.send_bytes(latest_image)
                except WebSocketDisconnect:
                    viewer_connections.remove(viewer)

    except WebSocketDisconnect:
        print("📷 Camera disconnected")

@app.websocket("/ws/viewer")
async def viewer_ws(websocket: WebSocket):
    await websocket.accept()
    print("👁️ Viewer connected")
    viewer_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        print("👁️ Viewer disconnected")
        viewer_connections.remove(websocket)
