import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import albumentations as Aq

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_1.model_odlocanja.model import obdelaj_sliko
from Stranski_model.stranski_mode import obdelaj_sliko_model_2
from Stranski_model.stranski_mode import obdelaj_sliko_model_2_1

video1_path = 'UI_for_ai/Video_009_25_4_2025.mp4'
video2_path = 'UI_for_ai/Video_009_25_4_2025.mp4'

important_labels = {
    "Prosto_parkirno_mesto": 1,
    "Družinsko_parkiranje": 2,
    "Električno_parkiranje": 3,
    "Invalidsko_parkiranje": 4
}

priority_order = [
    "Prosto_parkirno_mesto",
    "Družinsko_parkiranje",
    "Električno_parkiranje",
    "Invalidsko_parkiranje"
]

messages_video1 = [
    {"text": "Looking for parking...", "icon": "question"},
    {"text": "Free parking on right", "icon": "check"},
    {"text": "Family parking available", "icon": "check"},
    {"text": "EV parking available", "icon": "check"},
    {"text": "Disabled parking available", "icon": "check"},
]

messages_video2 = [
    {"text": "No obstacles", "icon": "check"},
    {"text": "Obstacle near", "icon": "question"},
    {"text": "Obstacle is close!", "icon": "question"},
    {"text": "STOP!", "icon": "error"}
]

# GUI functions removed
def play_sound_async(wav_file):
    pass

def draw_message_box(*args, **kwargs):
    pass  # Also disabled GUI drawing

value_switch = False

def play_videos_with_switch(video1_path, video2_path):
    ProcessedPic = 0
    global value_switch

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening one of the video files")
        return

    while True:
        cap = cap2 if value_switch else cap1
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        if not value_switch:
            annotated_frame_1, label_model = obdelaj_sliko_model_2(frame, 0.5)
            annotated_frame_2, _ = obdelaj_sliko_model_2_1(frame, 0.5)
        else:
            annotated_frame_1, label_model, danger_level = obdelaj_sliko(frame, 0.5)
            annotated_frame_2, _, _ = obdelaj_sliko(frame, 0.5)

        if value_switch:
            danger_level = int(np.clip(danger_level, 0, len(messages_video2) - 1))
            print("Danger level:", danger_level)
            msg = messages_video2[danger_level]
        else:
            important_label_found = None
            for label in priority_order:
                if label in label_model:
                    important_label_found = label
                    break
            if important_label_found is None:
                msg = messages_video1[0]
            else:
                idx = important_labels[important_label_found]
                msg = messages_video1[idx]

        #print(f"[Frame {ProcessedPic}] {msg['icon'].upper()}: {msg['text']}")

        # Optional: Save output frame (example)
        # cv2.imwrite(f"output/frame_{ProcessedPic}.jpg", annotated_frame_1)

        ProcessedPic += 1
        time.sleep(0.04)  # simulate ~25 FPS

    cap1.release()
    cap2.release()

play_videos_with_switch(video1_path, video2_path)
