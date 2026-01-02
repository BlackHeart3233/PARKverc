# ws_server.py
import asyncio
import json
import websockets
import threading

connected_clients = set()
loop = None

async def handler(websocket):
    connected_clients.add(websocket)
    print("JS povezan")
    try:
        async for _ in websocket:
            pass
    finally:
        connected_clients.remove(websocket)
        print("JS odklopljen")

async def ws_main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WebSocket server na ws://localhost:8765")
        await asyncio.Future()  # run forever

def start_ws_server():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_main())

# Ta funkcija pošlje sporočilo JS-u
async def send_to_js_async(message):
    if connected_clients:
        data = json.dumps(message)
        await asyncio.gather(*[ws.send(data) for ws in connected_clients])

def send_to_js(message):
    asyncio.run_coroutine_threadsafe(send_to_js_async(message), loop)
