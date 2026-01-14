import WebSocket from 'ws';
import { WebSocketServer } from 'ws';

//zagon node server.js
const wss = new WebSocketServer({ port: 8080 });

let sensorSockets,webSockets;

wss.on('connection', socket => {

    socket.on('message', async (message) => {
        try {
            const text = message.toString();
            const [command, payload] = text.split(" ", 2);

            switch (command) {
                case "WEB":
                    webSockets = socket;
                    console.log('webSockets povezan');
                    return;
                case "SENZOR":
                    sensorSockets = socket;
                    console.log('sensor povezan');
                    return;
                case "KAMERA:":
                    const compressedBuffer = Buffer.from(message);
                    const res = await fetch("http://localhost:5000/obdelaj_sliko", {
                        method: "POST",
                        headers: { "Content-Type": "application/octet-stream" },
                        body: compressedBuffer
                    });
                    const json = await res.json();
                    console.log(JSON.stringify(json, null, 2))
                    if (!webSockets || webSockets.readyState !== WebSocket.OPEN) {
                            console.log("WEB ni povezan - preskakujem");
                            return;
                        }
                    webSockets.send(JSON.stringify({json}));


                    break;
                default:
                    socket.send("Neznan ukaz");
            }
            socket.on("close", () => {
                if (socket === webSockets) webSockets = null;
                if (socket === sensorSockets) sensorSockets = null;
                console.log("Client odklopljen");
            });

        } catch (err) {
            console.error(err);
            socket.send("Napaka pri obdelavi");
        }
    });
});
