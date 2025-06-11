import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import threading
import winsound
import torch
from torchvision import transforms
from torchvision.models import resnet18

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_1.model_odlocanja.model import obdelaj_sliko, izpisi_in_izlusci

video1_path = 'Video_007_25_4_2025.mp4'
video2_path = 'IMG_4905.mp4'
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
}


def nalozi_model_brez_predpone(model, pot_do_modela, prefix="backbone."):
    import torch

    # Naloži state_dict
    state_dict = torch.load(pot_do_modela, map_location=torch.device('cpu'))

    # Odstrani predpono iz imen ključev
    novi_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            nov_k = k[len(prefix):]  # odstrani "backbone."
        else:
            nov_k = k
        novi_state_dict[nov_k] = v

    # Naloži popravljene uteži v model
    model.load_state_dict(novi_state_dict)
    return model


def nalozi_model(pot_do_modela="offset_model.pth"):
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model = nalozi_model_brez_predpone(model, pot_do_modela)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    return model, transform

def najdi_offset_iz_oznaka(oznaka_podatki, width, height, top_point=None, bottom_point_left=None, bottom_point_right=None):
    if top_point is None or bottom_point_left is None or bottom_point_right is None:
        # Če ni podatkov o trikotniku, izračunaj offset glede na središče slike
        ref_x = width // 2
    else:
        base_center = (np.array(bottom_point_left) + np.array(bottom_point_right)) / 2
        midline_point = (base_center + np.array(top_point)) / 2
        ref_x = midline_point[0]

    offset = None
    min_dist = float('inf')

    for oznaka in oznaka_podatki:
        if oznaka["label"] == "Parking_line":
            x_center = oznaka["bbox"][0]
            dist = abs(x_center - ref_x)
            if dist < min_dist:
                min_dist = dist
                offset = x_center - ref_x

    if offset is None:
        offset = 0

    max_offset = 150
    if offset > max_offset:
        offset = max_offset
    elif offset < -max_offset:
        offset = -max_offset

    return offset




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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_width = 1280
    target_height = 480
    video_width = int(target_width * 0.75) - (2 * PADDING_LEFT)
    message_width = target_width - video_width - (2 * PADDING_LEFT)
    video_scale = video_width / original_width
    video_height = int(original_height * video_scale)
    frame_height = video_height + (2 * PADDING_TOP_BOTTOM)

    background = cv2.imread(background_path)
    if background is None:
        print(f"Error loading background image: {background_path}")
        return
    background_resized = cv2.resize(background, (target_width, frame_height))

    message_index = 0
    last_played_sound = -1

    if draw_curves:
        model, transform = nalozi_model()
        width, height = target_width, frame_height
        start1 = np.array([50, height - 50])
        end1 = np.array([350, 200])
        start2 = np.array([900, height - 50])
        end2 = np.array([600, 200])
        current_offset = 0
        step = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, results = obdelaj_sliko(frame, 0.64)
        oznaka_podatki = izpisi_in_izlusci(results)

        height, width = frame.shape[:2]
        offset = najdi_offset_iz_oznaka(oznaka_podatki, width, height)
        offset = offset  # brez minusa, kot želiš

        resized_frame = cv2.resize(annotated_frame, (video_width, video_height))

        padded_frame = background_resized.copy()
        padded_frame[PADDING_TOP_BOTTOM:PADDING_TOP_BOTTOM + video_height,
                     PADDING_LEFT:PADDING_LEFT + video_width] = resized_frame

        key = cv2.waitKey(25) & 0xFF

        if key in [ord(str(i)) for i in range(1, len(messages)+1)]:
            message_index = key - ord('1')

        current_message = messages[message_index]
        draw_message_box(padded_frame, current_message["text"], icon_type=current_message["icon"])

        if current_message["text"] == "Free parking on right" and last_played_sound != message_index:
            last_played_sound = message_index
            play_sound_async(ding_sound_path)
        elif current_message["text"] != "Free parking on right":
            last_played_sound = -1

        if draw_curves:
            if current_offset < offset:
                current_offset = min(current_offset + step, offset)
            elif current_offset > offset:
                current_offset = max(current_offset - step, offset)
        else:
            current_offset = offset

        if draw_curves:
            control1 = (start1 + end1) / 2 + np.array([-current_offset, 0])
            control2 = (start2 + end2) / 2 + np.array([-current_offset, 0])
            draw_bezier_curve(padded_frame, start1, control1, end1, (0, 255, 0))
            draw_bezier_curve(padded_frame, start2, control2, end2, (0, 0, 255))

            bottom_point_left = tuple(bezier_quad(start1, control1, end1, 0.0).astype(int))
            bottom_point_right = tuple(bezier_quad(start2, control2, end2, 0.0).astype(int))

            top_point_x = (bottom_point_left[0] + bottom_point_right[0]) // 2
            top_point_y = int(height * 0.30)
            top_point = (top_point_x, top_point_y)

            #triangle_cnt = np.array([top_point, bottom_point_left, bottom_point_right])
            #cv2.drawContours(padded_frame, [triangle_cnt], 0, (255, 255, 0), -1)

            offset = najdi_offset_iz_oznaka(oznaka_podatki, width, height, top_point, bottom_point_left,
                                            bottom_point_right)
        else:
            offset = najdi_offset_iz_oznaka(oznaka_podatki, width, height, (0, 0), (0, 0),
                                            (0, 0))  # če ni krivulj, lahko daš dummy vrednosti ali preurediš funkcijo

        cv2.imshow('CTkMessagebox Style with Arial Font', padded_frame)

        if key == ord('q'):
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

play_video(video1_path, messages_video1, draw_curves=False)  # Prvi video brez črt
play_video(video2_path, messages_video2, draw_curves=True)   # Drugi video z ukrivljenimi črtami
