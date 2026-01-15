
import WebSocket from "ws";
import fs from "fs";
import path from "path";

const IMAGE_DIR = "C:/Users/konjc/OneDrive - Univerza v Mariboru/PARKverc/MERITVE/Meritve_2/Video/Frames_video/IMG_4905";   // 📁 mapa s slikami
const DELAY_MS = 200;          //200 ms ≈ 5 FPS

const socket = new WebSocket("ws://localhost:8080");

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

socket.onopen = async () => {
    console.log("Povezan na WS server");
    socket.send("SENZOR");
    let files = fs.readdirSync(IMAGE_DIR);
    files = files.filter(f =>
        f.endsWith(".jpg") || f.endsWith(".png") || f.endsWith(".jpeg")
    );
    files.sort();
    console.log("Najdenih slik:", files.length);
    for (const file of files) {
        const filePath = path.join(IMAGE_DIR, file);
        console.log("Pošiljam:", file);
        const imageBuffer = fs.readFileSync(filePath);
        const res = await fetch("http://127.0.0.1:8000/kompresija", {
            method: "POST",
            headers: {
                "Content-Type": "application/octet-stream"
            },
            body: imageBuffer
        });
        if (!res.ok) {
            console.error("Napaka pri kompresiji:", res.status);
            continue;
        }

        const compressedArrayBuffer = await res.arrayBuffer();
        const compressedBuffer = Buffer.from(compressedArrayBuffer);

        const base64 = compressedBuffer.toString("base64");
        socket.send(JSON.stringify({
            type: "KAMERA",
            data: base64,
            filename: file
        }));
        await sleep(DELAY_MS);
    }

    console.log("Vse slike poslane");
};

socket.onmessage = ({ data }) => {
    console.log("Odgovor strežnika:", data);
};

socket.onerror = err => {
    console.error("WS error:", err);
};


