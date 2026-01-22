import asyncio
import websockets

WS_URL = "ws://localhost:8000/ws/rotate"

async def main():
    async with websockets.connect(WS_URL) as ws:
        print("povezan na strežnik.")
        print("vnesi stopnje (e.g. 180, -540). Ctrl+C za izhod.")

        while True:
            deg = input("> ").strip()
            if not deg:
                continue

            await ws.send(deg)
            reply = await ws.recv()
            print("Server:", reply)

if __name__ == "__main__":
    asyncio.run(main())
