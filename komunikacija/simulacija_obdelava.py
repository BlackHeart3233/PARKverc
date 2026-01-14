from ultralytics import YOLO
import cv2
from typing import Union
from fastapi import FastAPI, Request, Response
import kompresija_FLOCIC
import numpy as np
from PIL import Image
import dekompresija__FLOCIC as dekompresija
import os
import time
from bitstring import BitArray
import threading
import cv2
import numpy as np

#_dummy = np.zeros((8,8), np.int32)
#kompresija_FLOCIC.kompresija(_dummy)

app = FastAPI()

#python -m uvicorn simulacija_obdelava:app --port 5000 --reload

#VZRATNI MODEL
model = YOLO("https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt")
PARKING_LINE_CLASS_ID = 7

@app.post("/obdelaj_sliko")
async def obdelaj_sliko(request: Request):
    data = {"lines": []}
    img_bytes = await request.body()
    if not img_bytes:
        return data
    B = BitArray(img_bytes)
    gray = dekompresija.Decompress(B)
    if gray is None:
        print("DECOMPRESS ERROR:")
        return data
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    os.makedirs("debug", exist_ok=True)
    filename = f"debug/debug_{int(time.time()*1000)}.png"
    cv2.imwrite(filename, img)
    result = model(img, verbose=False, conf=0.6)[0]
    if result.obb is None:
        return data
    for i, obb_data_row in enumerate(result.obb.xywhr):
        cls_id = result.obb.cls[i].item()
        if cls_id != 7:
            continue  # samo za parkirne črte

        x_center = obb_data_row[0].item()
        y_center = obb_data_row[1].item()
        width = obb_data_row[2].item()
        height = obb_data_row[3].item()
        angle = obb_data_row[4].item()

        # Rotirani bounding box
        box = ((x_center, y_center), (width, height), angle * 180 / np.pi)
        box_pts = cv2.boxPoints(box)
        box_pts = np.intp(box_pts)

        # Izpiši koordinate v terminal
        coords_list = box_pts.tolist()
        line_data = {
            "corners": [
                {"x": int(x), "y": int(y)} for x, y in coords_list
            ]
        }
        data["lines"].append(line_data)
    return data

@app.post("/kompresija")
async def kompresija(request: Request):
    raw = await request.body()
    img = cv2.imdecode(
        np.frombuffer(raw, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise ValueError("cv2.imdecode failed")

    height, width = img.shape
    P = img.astype(np.int32)
    B = kompresija_FLOCIC.kompresija(P)
    return Response(
        content=B.tobytes(),
        media_type="application/octet-stream"
    )

   
#stranski model  
