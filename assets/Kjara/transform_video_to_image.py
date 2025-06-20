import cv2
import os

"""
Pretvori video v slike pri določeni hitrosti frame-ov.

Parametri:
    video_path (str)
    output_dir (str)
    frames_per_second (int, optional) – privzeto 3
"""

def extract_frames(video_path, output_dir, frames_per_second=0):
    try:
        # Ustvari izhodno mapo
        os.makedirs(output_dir, exist_ok=True)

        # Odpri video
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise IOError("Ne morem odpret videja")

        # FPS iz videa, če uporabnik ne poda svoje vrednosti
        if frames_per_second == 0:
            frames_per_second = video.get(cv2.CAP_PROP_FPS)

        frame_interval_ms = 1000 / frames_per_second
        current_time_ms = 0
        saved_frame_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            current_frame_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)

            if current_frame_time_ms >= current_time_ms:
                saved_frame_count += 1
                output_path = os.path.join(output_dir, f"frame_{saved_frame_count}.jpg")
                cv2.imwrite(output_path, frame)
                current_time_ms += frame_interval_ms

        #Poizkusi še shraniti zadnji frame, če ga ni v zanki
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if saved_frame_count < total_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = video.read()
            if ret:
                saved_frame_count += 1
                output_path = os.path.join(output_dir, f"frame_{saved_frame_count}.jpg")
                cv2.imwrite(output_path, frame)
                print(f"Dodan še zadnji frame: frame_{saved_frame_count}.jpg")

        video.release()
        print(f"Shranjenih {saved_frame_count} frameov v {output_dir}")

    except Exception as e:
        print(f"Napaka: {e}")

def nasloviVideo (path):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = []

    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in video_extensions:
            video_files.append(filename)
    return video_files

if __name__ == "__main__":
    pathFolder = r"C:/Users/kjara/OneDrive - Univerza v Mariboru/PARKverc/MERITVE/Testne_meritve/Video"
    video_files = nasloviVideo(pathFolder)
    for videoName in video_files:
        video_path = os.path.join(pathFolder, videoName)
        output_dir = os.path.join(pathFolder, "Frames_video", videoName[:-4])
        extract_frames(video_path, output_dir, 25)

