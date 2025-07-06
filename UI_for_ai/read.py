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
global selected_option

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_1.model_odlocanja.model import obdelaj_sliko, izpisi_in_izlusci
from Stranski_model.stranski_mode import obdelaj_sliko_model_2
from Stranski_model.stranski_mode import obdelaj_sliko_model_2_1

#video1_path =['UI_for_ai/Video_004_25_4_2025.mp4','UI_for_ai/Video_009_25_4_2025.mp4','UI_for_ai/Video_005_25_4_2025.mp4']
#video2_path =['UI_for_ai/Video_004_25_4_2025.mp4','UI_for_ai/Video_009_25_4_2025.mp4','UI_for_ai/Video_005_25_4_2025.mp4']

video1_path = [
    "UI_for_ai/Video_004_25_4_2025.mp4",
    "UI_for_ai/Video_009_25_4_2025.mp4",
    "UI_for_ai/Video_005_25_4_2025.mp4"
]

video2_path = [
    "UI_for_ai/IMG_4905.mp4",
    "UI_for_ai/IMG_4898.mp4",
    "UI_for_ai/IMG_4902.mp4"
]

dropdown_options = ["Elektricno", "Druzinsko", "Invalidsko", "Navadno"]
selected_option = dropdown_options[0]
dropdown_open = False


background_path = r"UI_for_ai/background_l.jpg"
arial_path = 'UI_for_ai/ARIAL.TTF'
ding_sound_path = 'UI_for_ai/ding.mp3'

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
    {"text": "Iskanje parkirišča...", "icon": "question"},
    {"text": "Prosto parkirišče na desni", "icon": "check"},
    {"text": "Na voljo družinsko parkirišče", "icon": "check"},
    {"text": "Na voljo parkirišče za električna vozila", "icon": "check"},
    {"text": "Na voljo parkirišče za invalide", "icon": "check"},
]


messages_video2 = [
    {"text": "Ni ovir", "icon": "check"},
    {"text": "Ovira v bližini", "icon": "question"},
    {"text": "Ovira je zelo blizu!", "icon": "question"},
    {"text": "USTAVI!", "icon": "error"}
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
    global value_switch, dropdown_open, selected_option
    total_width, total_height, padding_left_right, padding_top_bottom = param
    button_x = (total_width - button_width) // 2
    button_y = total_height - padding_top_bottom - 10  # same as in main loop

    if event == cv2.EVENT_LBUTTONDOWN:
        # Switch button
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            value_switch = not value_switch
            dropdown_open = False

        # Dropdown button location and logic when value_switch is False (like in drawing)
        if not value_switch:
            dropdown_x = button_x + 500
            dropdown_y = button_y - 220
            dropdown_width = 200
            dropdown_height = 40

            # Toggle dropdown
            if dropdown_x <= x <= dropdown_x + dropdown_width and dropdown_y <= y <= dropdown_y + dropdown_height:
                dropdown_open = not dropdown_open

            # Selecting option
            if dropdown_open:
                for i, option in enumerate(dropdown_options):
                    option_y = dropdown_y + (i + 1) * dropdown_height
                    if dropdown_x <= x <= dropdown_x + dropdown_width and option_y <= y <= option_y + dropdown_height:
                        selected_option = option
                        dropdown_open = False



value_switch = False
button_width = 200
button_height = 47

#----------------------------------------------------------------------
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
        if isinstance(oznaka, dict):  # Preverimo, da je oznaka slovar
            if oznaka.get("label") == "Parking_line":
                x_center = oznaka.get("bbox", [0])[0]
                dist = abs(x_center - ref_x)
                if dist < min_dist:
                    min_dist = dist
                    offset = x_center - ref_x
        else:
            # Če oznaka ni slovar, jo lahko preskočimo ali dodamo kakšno drugo logiko
            pass

    if offset is None:
        offset = 0

    max_offset = 150
    if offset > max_offset:
        offset = max_offset
    elif offset < -max_offset:
        offset = -max_offset

    #print("OFFSET JE: ", offset)
    return offset

def bezier_quad(p0, p1, p2, t): #to je za one krivulje ko se izrisujejo
    #izračuna točko na kvadratni Bezierjevi krivulji za parameter t
    p01 = (1 - t) * p0 + t * p1
    p12 = (1 - t) * p1 + t * p2
    return (1 - t) * p01 + t * p12

def draw_bezier_curve(img, start, end, control, color=(255, 255, 255), thickness=2, num_points=100):
    points = []
    for t in np.linspace(0, 1, num_points):
        x = int((1 - t)**2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0])
        y = int((1 - t)**2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1])
        points.append((x, y))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, thickness)

def najdi_offset_po_y(oznaka_podatki):
    najvisja_y = float('inf')
    najvisja_x = None
    
    for oznaka in oznaka_podatki:
        if isinstance(oznaka, dict) and oznaka.get("label") == "Parking_line":
            bbox = oznaka.get("bbox", None)
            if bbox and len(bbox) >= 2:
                x, y = bbox[0], bbox[1]
                if y < najvisja_y:
                    najvisja_y = y
                    najvisja_x = x
                    
    if najvisja_x is None:
        return 0

    offset = int(np.sqrt(najvisja_x**2 + najvisja_y**2))
    #print(f"Najvišja Parking_line oznaka: x={najvisja_x}, y={najvisja_y}, offset={offset}")
    return offset


def compute_flexible_control_offset(offset, min_shift=100):
    if offset == 0:
        return np.array([0, 0])

    direction = -1 if offset > 0 else 1
    base_shift = max(abs(offset), min_shift)

    nonlinear_shift = base_shift ** 1.5

    max_shift = 200
    shift = min(nonlinear_shift, max_shift)

    return np.array([direction * shift, 0])


#----------------------------------------------------------------------

global result

def play_videos_with_switch(video1_path, video2_path, draw_curves=False):
    ProcessedPic = 0
    global value_switch

    for i in range(3):
        cap1 = cv2.VideoCapture(video1_path[i])
        cap2 = cv2.VideoCapture(video2_path[i])
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

        important_labels = {
            "Prosto_parkirno_mesto": 1,
            "Družinsko_parkiranje": 2,
            "Električno_parkiranje": 3,
            "Invalidsko_parkiranje": 4
        }

        base_order = [
            "Prosto_parkirno_mesto",
            "Družinsko_parkiranje",
            "Električno_parkiranje",
            "Invalidsko_parkiranje"
        ]

        option_map = {
            "Elektricno": "Električno_parkiranje",
            "Druzinsko": "Družinsko_parkiranje",
            "Invalidsko": "Invalidsko_parkiranje",
            "Navadno": "Prosto_parkirno_mesto"
        }

        selected_label = option_map.get(selected_option, "Prosto_parkirno_mesto")

        priority_order = [selected_label] + [label for label in base_order if label != selected_label]

        print("Priority: ",priority_order)

        current_offset = 0


        while True:
            cap = cap2 if value_switch else cap1
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            combined_frame = background_resized.copy()
            oznaka_podatki = {}

            calculated_danger = 0

            if not value_switch:
                annotated_frame_1, label_model = obdelaj_sliko_model_2(frame, 0.5)
                annotated_frame_2, _ = obdelaj_sliko_model_2_1(frame, 0.5)
            else:
                annotated_frame_1, results, reported_danger, slika_z_poudarjenimi_crtami = obdelaj_sliko(frame, 0.5)
                annotated_frame_1 = slika_z_poudarjenimi_crtami
                calculated_danger = reported_danger

                annotated_frame_2,_,_,_ = obdelaj_sliko(frame, 0.5)
                oznaka_podatki = izpisi_in_izlusci(results)

            resized_frame_left = cv2.resize(annotated_frame_1, (video_width_1, video_height_1))
            resized_frame_right = cv2.resize(annotated_frame_2, (video_width_2, video_height_2))

            combined_frame[
                padding_top_bottom:padding_top_bottom + resized_frame_left.shape[0],
                padding_left_right:padding_left_right + resized_frame_left.shape[1]
            ] = resized_frame_left

            combined_frame[
                padding_top_bottom:padding_top_bottom + resized_frame_right.shape[0],
                padding_left_right + resized_frame_left.shape[1] + padding_between:
                padding_left_right + resized_frame_left.shape[1] + padding_between + resized_frame_right.shape[1]
            ] = resized_frame_right

            selected_label = option_map.get(selected_option, "Prosto_parkirno_mesto")
            priority_order = [selected_label] + [label for label in base_order if label != selected_label]
            print("Priority: ",priority_order)

                

            if value_switch:
                # Inicializacija danger_level na neko privzeto vrednost, npr. 0
                danger_level = calculated_danger or 0
                #print("danger lvl je: ", danger_level)
                msg = messages_video2[danger_level]
                #print("msg je: ", msg)
            else:
                danger_level = 0  # tudi tu dodaš default, da ne pride do napake
                danger_level = calculated_danger or 0

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
                height=40
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

            if not value_switch:
                dropdown_x = button_x + 500
                dropdown_y = button_y - 220
                dropdown_width = 200
                dropdown_height = 40

                # Draw the dropdown button
                cv2.rectangle(combined_frame, (dropdown_x, dropdown_y),
                            (dropdown_x + dropdown_width, dropdown_y + dropdown_height),
                            (70, 70, 70), -1)
                cv2.putText(combined_frame, selected_option,
                            (dropdown_x + 10, dropdown_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # If dropdown is open, draw options
                if dropdown_open:
                    for i, option in enumerate(dropdown_options):
                        option_y = dropdown_y + (i + 1) * dropdown_height
                        cv2.rectangle(combined_frame, (dropdown_x, option_y),
                                    (dropdown_x + dropdown_width, option_y + dropdown_height),
                                    (50, 50, 50), -1)
                        cv2.putText(combined_frame, option,
                                    (dropdown_x + 10, option_y + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            

                    # ------ BEZIER KRIVULJE ------
            if value_switch:
                if draw_curves:
                    width, height = total_width, total_height
                    start1 = np.array([50, height - 111])
                    end1 = np.array([350, 300])
                    start2 = np.array([900, height - 111])
                    end2 = np.array([600, 300])
                      # Če imaš offset izračunan, ga tukaj uporabi
                    step = 10

                    height, width = frame.shape[:2]
                    new_offset = najdi_offset_iz_oznaka(oznaka_podatki, width, height)


                    if current_offset < new_offset:
                        current_offset = min(current_offset + step, new_offset)
                    elif current_offset > new_offset:
                        current_offset = max(current_offset - step, new_offset)

                    control1 = np.round((start1 + end1) / 2 + np.array([-current_offset, 0])).astype(int)
                    control2 = np.round((start2 + end2) / 2 + np.array([-current_offset, 0])).astype(int)

                    draw_bezier_curve(combined_frame, start1, end1, control1, color=(255, 0, 0))
                    draw_bezier_curve(combined_frame, start2, end2, control2, color=(0, 0, 255))

                    bottom_point_left = tuple(bezier_quad(start1, control1, end1, 0.0).astype(int))
                    bottom_point_right = tuple(bezier_quad(start2, control2, end2, 0.0).astype(int))


                    top_point_x = (bottom_point_left[0] + bottom_point_right[0]) // 2
                    top_point_y = int(height * 0.30)
                    top_point = (top_point_x, top_point_y)

                    # Uporabi result, če je prazen, pa poskusi uporabiti label_model:

                    labels_for_offset = []
                    if oznaka_podatki and isinstance(oznaka_podatki, list) and len(oznaka_podatki) > 0:
                        first_item = oznaka_podatki[0]
                        if isinstance(first_item, dict):
                            labels_for_offset = first_item.get("labels", [])


                    if not labels_for_offset and not value_switch:
                        # Morda uporabi label_model, če ni result
                        labels_for_offset = label_model if label_model else []

                    offset = najdi_offset_iz_oznaka(labels_for_offset, width, height, top_point, bottom_point_left, bottom_point_right)
                else:
                    offset = najdi_offset_iz_oznaka(oznaka_podatki.get("labels", []), width, height, (0, 0), (0, 0), (0, 0))

            cv2.imshow(window_name, combined_frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break

            ProcessedPic += 1
            #publish.single(TOPIC1, ProcessedPic, hostname=BROKER)

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

play_videos_with_switch(video1_path, video2_path, draw_curves=True)
