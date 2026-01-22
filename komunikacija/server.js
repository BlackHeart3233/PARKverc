import WebSocket, { WebSocketServer } from "ws";



const wss = new WebSocketServer({ port: 8080 });

let sensorSocket = null;
let webSocket = null;

console.log("WebSocket server teče na ws://localhost:8080");

wss.on("connection", socket => {

    console.log("Client povezan");
    socket.on("message", async (message) => {
        //console.log("message je: ", message)
        try {
            const text = message.toString();
            let obj;
            try {
                obj = JSON.parse(text);
            } catch {
                if (text === "WEB") {
                    webSocket = socket;
                    console.log("WEB povezan");
                    return;
                }

                if (text === "SENZOR") {
                    sensorSocket = socket;
                    console.log("SENZOR povezan");
                    return;
                }
                console.log("Neznan tekst:", text);
                return;
            }

            if (obj.type === "KAMERA") {
                if (!obj.data) {
                    console.log("KAMERA brez data");
                    return;
                }
                const compressedBuffer = Buffer.from(obj.data, "base64");

                console.log(
                    "Prejet KAMERA frame:",
                    compressedBuffer.length,
                    "bytes"
                );
                const res = await fetch("http://localhost:8000/obdelaj_sliko", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/octet-stream"
                    },
                    body: compressedBuffer
                });

                let result;
                try {
                    result = await res.json();
                } catch {
                    result = {
                        error: "FastAPI ni vrnil JSON",
                        status: res.status
                    };
                }

            if (webSocket && webSocket.readyState === WebSocket.OPEN) {
                webSocket.send(JSON.stringify({
                json: result.json
                }));
                webSocket.send(JSON.stringify({ type: "IMAGE", image: result.image }));
            } else {
                console.log("WEB ni povezan");
            }

                return;
            }

            console.log("Neznan JSON:", obj);

        } catch (err) {
            console.error("Napaka WS message:", err);
        }
    });

    socket.on("close", () => {
        if (socket === webSocket) {
            webSocket = null;
            console.log("WEB odklopljen");
        }
        if (socket === sensorSocket) {
            sensorSocket = null;
            console.log("SENZOR odklopljen");
        }
    });
});
