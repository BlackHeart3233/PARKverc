import WebSocket from 'ws';
import sharp from 'sharp';
import fs from 'fs';
import sizeOf from 'image-size';


//tole je sam za testirat delovanje 
//zagon je node client.js mores pa prvo met przgan server.js

const socket = new WebSocket('ws://localhost:8080');
socket.onopen = async () => {
    console.log("Povezan na WS server");
    socket.send("SENZOR");
    const imageBuffer = fs.readFileSync("test.jpg");
    const res = await fetch("http://127.0.0.1:5000/kompresija", {
    method: "POST",
    headers: {
        "Content-Type": "application/octet-stream"
    },
    body: imageBuffer
    });
    //Response → ArrayBuffer → Buffer
    const compressedArrayBuffer = await res.arrayBuffer();
    const compressedBuffer = Buffer.from(compressedArrayBuffer);
    //Buffer → base64
    //const base64 = compressedBuffer.toString('base64');
    socket.send(`KAMERA: ${compressedBuffer}`);
};

socket.onmessage = ({ data }) => {
    console.log('Prijel sem:', data);
};

socket.onerror = err => {
    console.error('WebSocket error', err);
};
