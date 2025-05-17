import json
import os


name_to_id = {
    "Ball_2025_03_30_20_54": 0,
    "Bicycle_2025_03_30_20_54": 1,
    "Bike_2025_03_30_20_54": 2,
    "Car_2025_03_30_20_54": 3,
    "Cart_2025_03_30_20_54": 4,
    "Curb_2025_03_30_20_54": 5,
    "Human_2025_03_30_20_54": 6,
    "Pole_2025_03_30_20_54": 7,
    "Ramp_2025_03_30_20_54": 8,
    "Parking_line_2025_03_30_20_54": 9,
    "Electric_car_2025_04_28_10_50": 10,
    "Family_car_2025_04_28_10_50": 11,
    "Handicapped_parking_2025_04_28_10_50": 12
}

videoName = "Video_007_25_4_2025.mp4"
pathJSONMINI = "./assets/Kjara/labels_video/Video_007_25_4_2025_MINI_JSON.json"
with open(pathJSONMINI, "r") as file:
    data_all = json.load(file)

# Find the index of the video you're interested in
video_index = None
for idx, entry in enumerate(data_all):
    if videoName in entry["video"]:
        video_index = idx
        break
if video_index is None:
    raise ValueError(f"Video {videoName} not found in JSON")

os.makedirs("./test", exist_ok=True)

data = data_all[video_index]
stevilo_framov = data["box"][0]['framesCount']
for i in range(1,stevilo_framov+1):
    file = open(f"./test/{i}_frame.text", "w")
    file.close()



for k in range(len(data["box"])): #ponovi se tolikokrat kolikor je obejktov
    firstFrame = data["box"][k]['sequence'][0]['frame']
    lastFrame = data["box"][k]['sequence'][-1]['frame']
    class_id = data["box"][k]['labels'][0]
    currentFrame = firstFrame
    nextFrame = data["box"][k]['sequence'][1]['frame']

    if len(data["box"][k]['sequence']) > 1:
        nextFrame = data["box"][k]['sequence'][1]['frame']
    else:
        nextFrame = lastFrame

    current_output = ""
    visible = data["box"][k]['sequence'][0]['enabled']
    #<class_id> <x_center_rel> <y_center_rel> <width_rel> <height_rel>

    current_frameCounter = firstFrame
    first = True
    counterSequence = 0 #len(son_data["box"][k]['sequence'])
    for i in range(lastFrame-firstFrame+1):
        if first or current_frameCounter == nextFrame:
            x = data["box"][k]['sequence'][counterSequence]['x']
            y = data["box"][k]['sequence'][counterSequence]['y']
            width = data["box"][k]['sequence'][counterSequence]['width']
            height = data["box"][k]['sequence'][counterSequence]['height']
            x_center_rel = (x + width / 2) / 100
            y_center_rel = (y + height / 2) / 100
            width_rel = width / 100
            height_rel = height / 100
            counterSequence+=1
            current_output = f"{name_to_id[class_id]} {x_center_rel:.4f} {y_center_rel:.4f} {width_rel:.4f} {height_rel:.4f}\n"
            if nextFrame != lastFrame and not first:
                nextFrame = data["box"][k]['sequence'][counterSequence]['frame']
                currentFrame = data["box"][k]['sequence'][counterSequence]['frame']
                visible = data["box"][k]['sequence'][counterSequence]['enabled']
            first = False
        if visible:
            with open(f"./test/{current_frameCounter}_frame.text","a") as f:
                f.write(current_output)
        current_frameCounter +=1
     

  

