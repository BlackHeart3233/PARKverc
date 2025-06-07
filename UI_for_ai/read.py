import cv2
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import threading
import winsound
import albumentations as A

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_1.model_odlocanja.model import obdelaj_sliko
from Stranski_model.stranski_mode import obdelaj_sliko_model_2
from Stranski_model.stranski_mode import obdelaj_sliko_model_2_1

#Zaenkrat sem jaz samo svoje videe dala not, to se bo še posodobilo
video1_path = 'UI_for_ai/Video_009_25_4_2025.mp4'
video2_path = 'UI_for_ai/Video_006_28_3_2025.mp4'
background_path = r'UI_for_ai/background.jpg'
arial_path = 'UI_for_ai/ARIAL.TTF'
ding_sound_path = 'UI_for_ai/ding.mp3'

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


messages_video1 = [ 
    {"text": "Looking for parking...", "icon": "question"},
    {"text": "Free parking on right", "icon": "check"},
    {"text": "Handicap parking available", "icon": "check"},
    {"text": "Family parking available", "icon": "check"},
    {"text": "Private parking available", "icon": "check"},
]

messages_video2 = [
    {"text": "No obstacles", "icon": "check"},
    {"text": "Obstacle near", "icon": "question"},
    {"text": "Obstacle is close!", "icon": "question"},
    {"text": "STOP!", "icon": "error"}
]


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


value_switch = False

button_width = 200
button_height = 50

def mouse_callback(event, x, y, flags, param):
    global value_switch
    total_width, total_height, padding_left_right, padding_top_bottom = param
    
    button_x = (total_width - button_width) // 2
    button_y = total_height - padding_top_bottom + 10  
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            value_switch = not value_switch
            print(f"VALUE SWITCH toggled: {value_switch}")


def play_videos_with_switch(video1_path, video2_path):
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
    video_width_1 = int(width1 * scale_factor)
    video_height_1 = int(height1 * scale_factor)
    video_width_2 = int(width2 * scale_factor)
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

    message_display_time = 4  
    start_time = time.time()
    
    idx1 = 0
    # Za Bezierjeve krivulje (če jih rišemo)
    '''
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
        '''

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

    while True:
        cap = cap2 if value_switch else cap1
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame after reset")
                break
        
        combined_frame = background_resized.copy()

        if not value_switch:
            annotated_frame_1, label_model = obdelaj_sliko_model_2(frame, 0.5)
            annotated_frame_2, _ = obdelaj_sliko_model_2_1(frame, 0.5)
        else:
            annotated_frame_1, label_model = obdelaj_sliko(frame, 0.5)
            annotated_frame_2, _ = obdelaj_sliko(frame, 0.5)

        if not value_switch:
            resized_frame_left = cv2.resize(annotated_frame_1, (video_width_2, video_height_2))
            resized_frame_right = cv2.resize(annotated_frame_2, (video_width_2, video_height_2))
        else:
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

        important_label_found = None
        for label in priority_order:
            if label in label_model:
                important_label_found = label
                break

        if important_label_found is None:
            msg = messages_video1[idx1]
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

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
#annotated_frame, results, danger_level = obdelaj_sliko(frame, 0.64)





play_videos_with_switch(video1_path, video2_path)
'''        current_message = messages[danger_level]

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
'''