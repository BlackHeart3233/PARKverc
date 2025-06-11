import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import threading
import winsound
import albumentations as Aq
#import paho.mqtt.publish as publish

BROKER = "10.0.0.1"
TOPIC1 = "spo/procesirane_slike"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_1.model_odlocanja.model import obdelaj_sliko
from Stranski_model.stranski_mode import obdelaj_sliko_model_2
from Stranski_model.stranski_mode import obdelaj_sliko_model_2_1

video1_path = './Video_009_25_4_2025.mp4'
video2_path = './IMG_4911.mp4'
background_path = r'./background.jpg'
arial_path = './ARIAL.TTF'
ding_sound_path = './ding.mp3'

PADDING_LEFT = 20
PADDING_TOP_BOTTOM = 20
BG_COLOR = (30, 30, 30, 220)
TEXT_COLOR = (255, 255, 255)

ICON_COLORS = {
    "check": (0, 255, 0),
    "error": (0, 0, 255),
    "warning": (0, 255, 255),
    "question": (255, 165, 0)
}

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
    {"text": "STOP! Obstacle in the way", "icon": "error"}
]

def draw_message_box(frame, message, icon_type="info", x=960, y=200, width=300, height=80):
    overlay = frame.copy()
    alpha = BG_COLOR[3] / 255.0

    cv2.rectangle(overlay, (x, y), (x + width, y + height), BG_COLOR[:3], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    icon_color = ICON_COLORS.get(icon_type, (255, 255, 255))
    cv2.circle(frame, (x + 30, y + height // 2), 15, icon_color, -1)

    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(arial_path, 20)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x + 60
    text_y = y + (height - text_height) // 2
    draw.text((text_x, text_y), message, font=font, fill=TEXT_COLOR)
    frame[:] = np.array(pil_image)

def play_sound_async(wav_file):
    threading.Thread(target=winsound.PlaySound, args=(wav_file, winsound.SND_FILENAME | winsound.SND_ASYNC), daemon=True).start()

def mouse_callback(event, x, y, flags, param):
    global value_switch
    total_width, total_height, padding_left_right, padding_top_bottom = param
    button_x = (total_width - button_width) // 2
    button_y = total_height - padding_top_bottom + 10
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            value_switch = not value_switch
            print(f"VALUE SWITCH toggled: {value_switch}")

value_switch = False
button_width = 200
button_height = 50

def play_videos_with_switch(video1_path, video2_path):
    ProcessedPic = 0
    global value_switch

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening one of the video files")
        return

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = 0.5
    video_width_1 = 500
    video_height_1 = int(height1 * scale_factor)
    video_width_2 = 500
    video_height_2 = int(height2 * scale_factor)

    padding_between = 30
    padding_top_bottom = 40
    padding_left_right = 40
    total_width = video_width_1 + video_width_2 + padding_between + 2 * padding_left_right
    total_height = max(video_height_1, video_height_2) + 2 * padding_top_bottom + 70

    background = cv2.imread(background_path)
    if background is None:
        print(f"Error loading background image: {background_path}")
        return
    background_resized = cv2.resize(background, (total_width, total_height))

    window_name = 'ParkVERC UI'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=(total_width, total_height, padding_left_right, padding_top_bottom))

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

    skip_frames = 20
    frame_counter = 0

    while True:
        frame_counter += 1

        if frame_counter % skip_frames == 0:
            frame_counter = 0
            continue
        cap = cap2 if value_switch else cap1
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        combined_frame = background_resized.copy()

        if not value_switch:
            annotated_frame_1, label_model = obdelaj_sliko_model_2(frame, 0.5)
            annotated_frame_2, _ = obdelaj_sliko_model_2_1(frame, 0.5)
        else:
            annotated_frame_1, label_model, danger_level = obdelaj_sliko(frame, 0.5, False)
            annotated_frame_2, _, _ = obdelaj_sliko(frame, 0.5, True)

        resized_frame_left = cv2.resize(annotated_frame_1, (video_width_1, video_height_1))
        resized_frame_right = cv2.resize(annotated_frame_2, (video_width_1, video_height_1))

        combined_frame[
            padding_top_bottom:padding_top_bottom + resized_frame_left.shape[0],
            padding_left_right:padding_left_right + resized_frame_left.shape[1]
        ] = resized_frame_left

        combined_frame[
            padding_top_bottom:padding_top_bottom + resized_frame_right.shape[0],
            padding_left_right + resized_frame_left.shape[1] + padding_between:
            padding_left_right + resized_frame_left.shape[1] + padding_between + resized_frame_right.shape[1]
        ] = resized_frame_right

        if value_switch:
            danger_level = int(np.clip(danger_level, 0, len(messages_video2) - 1))
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

        draw_message_box(
            combined_frame,
            msg["text"],
            icon_type=msg["icon"],
            x=padding_left_right,
            y=padding_top_bottom + resized_frame_left.shape[0] + 10,
            width=resized_frame_left.shape[1],
            height=50
        )

        button_x = (total_width - button_width) // 2
        button_y = total_height - padding_top_bottom - 10
        button_color = (0, 150, 0) if value_switch else (50, 50, 50)
        cv2.rectangle(combined_frame, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "SWITCH"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2
        cv2.putText(combined_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, combined_frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

        ProcessedPic += 1
        #publish.single(TOPIC1, ProcessedPic, hostname=BROKER)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

play_videos_with_switch(video1_path, video2_path)
