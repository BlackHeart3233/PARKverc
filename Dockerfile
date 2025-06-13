FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1-mesa-glx \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install \
    opencv-python-headless \
    numpy \
    matplotlib \
    pillow \
    ultralytics \
    albumentations \
    paho-mqtt

CMD ["python", "UI_for_ai/main.py"]
