from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import cv2

from decompression import razsiri_sliko_bytes  # <-- HERE

app = FastAPI()
clients = set()

@app.get("/")
def index():
    return HTMLResponse("""
    <html><body>
    <img id="img" width="640"/>
    <script>
      const ws = new WebSocket("ws://localhost:8000/ws/browser");
      ws.binaryType = "arraybuffer";
      ws.onmessage = e => {
        const blob = new Blob([e.data], {type: "image/jpeg"});
        img.src = URL.createObjectURL(blob);
      };
    </script>
    </body></html>
    """)

@app.websocket("/ws/browser")
async def ws_browser(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        clients.discard(ws)

@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            frame_bgr = razsiri_sliko_bytes(data)          # <-- decompress to numpy image
            ok, jpg = cv2.imencode(".jpg", frame_bgr)      # browser needs jpeg
            if not ok:
                continue
            for c in list(clients):
                try:
                    await c.send_bytes(jpg.tobytes())
                except:
                    clients.discard(c)
    except:
        pass
