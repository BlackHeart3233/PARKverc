import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import threading
import winsound

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_1.model_odlocanja.model import obdelaj_sliko


#Zaenkrat sem jaz samo svoje videe dala not, to se bo še posodobilo
video1_path = 'Video_009_25_4_2025 (1).mp4'
video2_path = 'IMG_4902.mp4'
background_path = r'background.jpg'
arial_path = 'ARIAL.TTF'
ding_sound_path = 'ding.mp3'

#some aesthetic zadevice tule
PADDING_LEFT = 20
PADDING_TOP_BOTTOM = 20
BG_COLOR = (30, 30, 30, 220)
TEXT_COLOR = (255, 255, 255)

ICON_COLORS = { #tole so barve
    "check": (0, 255, 0), #zlena
    "error": (0, 0, 255), #rdeča
    "warning": (0, 255, 255), #rumena
    "question": (255, 165, 0) #oranžna
}

def draw_message_box(frame, message, icon_type="info", x=960, y=200, width=300, height=80):
    overlay = frame.copy() #za risanje messageboxa
    alpha = BG_COLOR[3] / 255.0 #to nastima da je slightly prozorno


    #nariše prozorni pravokotnik
    cv2.rectangle(overlay, (x, y), (x + width, y + height), BG_COLOR[:3], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    #določi barvo glede na icon_type pa rarše krogec
    icon_color = ICON_COLORS.get(icon_type, (255, 255, 255))
    cv2.circle(frame, (x + 30, y + height // 2), 15, icon_color, -1)


    #OpenCV matriko pretvori v PIL sliko, da lahko uporabimo PIL pisave (ImageFont).
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype(arial_path, 20) #proba naložit arial pisavo
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font) #prikaže besedilo z izbrano pisavo pa barvo
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x + 60
    text_y = y + (height - text_height) // 2
    draw.text((text_x, text_y), message, font=font, fill=TEXT_COLOR)
    frame[:] = np.array(pil_image) #Posodobljeno PIL sliko pretvori nazaj v OpenCV matriko

def play_sound_async(wav_file):     #Tole je zvočni signal za takrat ko bo naš model zaznal fraj parking :p
    threading.Thread(target=winsound.PlaySound, args=(wav_file, winsound.SND_FILENAME | winsound.SND_ASYNC), daemon=True).start()
    #pol sem misla naredit še zvočne signale za parking mode, ko se bližamo neki oviri da začne sam bolj na hitro piskat

def bezier_quad(p0, p1, p2, t): #to je za one krivulje ko se izrisujejo
    #izračuna točko na kvadratni Bezierjevi krivulji za parameter t
    return (1 - t)**2 * p0 + 2*(1 - t)*t * p1 + t**2 * p2

def draw_bezier_curve(img, p0, p1, p2, color):
    points = [] #ustvari seznam points za shranjevanje točk krivulje

    for t in np.linspace(0, 1, 100): #ustvari 100 točk
        pt = bezier_quad(p0, p1, p2, t) #izračuna pozicijo točke na kruvulji glede na t
        points.append(tuple(pt.astype(int)))

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, 3) #dejansko riše krivuljo pol

def play_video(video_path, messages, draw_curves=False):
    cap = cv2.VideoCapture(video_path) #odpira dat
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_width = 1280     #tole so dimenzije za celoten ui
    target_height = 480
    video_width = int(target_width * 0.75) - (2 * PADDING_LEFT)
    message_width = target_width - video_width - (2 * PADDING_LEFT)
    video_scale = video_width / original_width
    video_height = int(original_height * video_scale)

    frame_height = video_height + (2 * PADDING_TOP_BOTTOM)

    background = cv2.imread(background_path) #slika ozdaja lol
    if background is None:
        print(f"Error loading background image: {background_path}")
        return
    background_resized = cv2.resize(background, (target_width, frame_height))

    message_index = 0
    last_played_sound = -1

    # Za Bezierjeve krivulje (če jih rišemo)
    if draw_curves:
        width, height = target_width, frame_height
        start1 = np.array([50, height - 50]) #fiksna začetna točka 1. krivulje
        end1 = np.array([350, 200])#fiksna končna točka 1. krivulje
        start2 = np.array([900, height - 50]) #fiksna začetna točka 2. krivulje
        end2 = np.array([600, 200])#fiksna končna točka 2 krivulje
        control_offset1 = 0
        control_offset2 = 0
        max_offset = 150 #omejitec premika kontrolne otčne
        step = 30 #korak premika    PO MOZNOSTI SPREMINJAJ TOTE CREDNOSTI DA BO BOLJ IDEALNO

    while cap.isOpened(): #ko je video odprt
        ret, frame = cap.read() #prebere naslednji okvir iz videa
        if not ret:
            break

        annotated_frame, results, danger_level = obdelaj_sliko(frame, 0.64)

        resized_frame = cv2.resize(annotated_frame, (video_width, video_height)) #prilagaja velikosti. possibly problem za naprej

        padded_frame = background_resized.copy()
        padded_frame[ #aestheticsss
            PADDING_TOP_BOTTOM:PADDING_TOP_BOTTOM + video_height,
            PADDING_LEFT:PADDING_LEFT + video_width
        ] = resized_frame

        key = cv2.waitKey(25) & 0xFF #Čaka 25 ms, kar določa hitrost predvajanja videa. & 0xFF se uporablja za branje vrednosti tipke

        if key in [ord(str(i)) for i in range(1, len(messages)+1)]: #to je za kontroliranje sporočil s tipkami
            message_index = key - ord('1')

        current_message = messages[danger_level]

        print(f"Current message: {current_message['text']}") #to je za debuganje
        print(f"Current danger level: {danger_level}") #to je za debuganje

        draw_message_box(padded_frame, current_message["text"], icon_type=current_message["icon"])

        if current_message["text"] == "Free parking on right" and last_played_sound != message_index: #to aktivira sound effect
            last_played_sound = message_index
            play_sound_async(ding_sound_path)
        elif current_message["text"] != "Free parking on right":
            last_played_sound = -1

        if draw_curves:
            control1 = (start1 + end1) / 2 + np.array([control_offset1, 0])
            control2 = (start2 + end2) / 2 + np.array([control_offset2, 0])
            draw_bezier_curve(padded_frame, start1, control1, end1, (0, 255, 0))
            draw_bezier_curve(padded_frame, start2, control2, end2, (0, 0, 255))

            if key == ord('a'): #S TEM KONTROLIRAŠ KRIVULJE A POMENI LEVO
                control_offset1 = max(control_offset1 - step, -max_offset)
                control_offset2 = max(control_offset2 - step, -max_offset)
            elif key == ord('d'):#D POMENI DESNO
                control_offset1 = min(control_offset1 + step, max_offset)
                control_offset2 = min(control_offset2 + step, max_offset)

        cv2.imshow('CTkMessagebox Style with Arial Font', padded_frame)

        if key == ord('q'): #QUIT BUTTON
            break

    cap.release()

cv2.destroyAllWindows()

messages_video1 = [ #seznam sporočil, po možnosti jih spremeni
    {"text": "Looking for parking...", "icon": "question"},
    {"text": "Free parking on right", "icon": "check"},
    {"text": "Handicap parking available", "icon": "check"},
    {"text": "No parking available", "icon": "error"}
]

messages_video2 = [
    {"text": "No obstacles", "icon": "check"},
    {"text": "Obstacle near", "icon": "question"},
    {"text": "Obstacle is close!", "icon": "question"},
    {"text": "STOP!", "icon": "error"}
]

#play_video(video1_path, messages_video1, draw_curves=False)  # Prvi video brez črt
play_video(video2_path, messages_video2, draw_curves=True)   # Drugi video z ukrivljenimi črtami
