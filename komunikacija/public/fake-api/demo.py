from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await ws.accept()
    print("✅ Camera connected")

    try:
        while True:
            data = await ws.receive_bytes()
            print(f"📦 Received {len(data)} bytes")
    except Exception as e:
        print("❌ Camera disconnected:", e)

if __name__ == "__main__":
    uvicorn.run(
        "demo:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
