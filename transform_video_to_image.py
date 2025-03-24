import cv2
import os

"""
video to images converer at a specific frame rate

params:
    video_path (str)
    output_dir (str)
    frames_per_second (int, optional) defaults to 3.
"""


def extract_frames(video_path, output_dir, frames_per_second=3):
    try:

        # create output dir
        os.makedirs(output_dir, exist_ok=True)

        # poen vide
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise IOError("Ne morem odpret videja")

        # get fps
        fps = video.get(cv2.CAP_PROP_FPS)

        # frame intervals in milliseconds
        frame_interval_ms = 1000 / frames_per_second

        current_time_ms = 0
        saved_frame_count = 0

        while True:
            ret, frame = video.read()

            # end of video
            if not ret:
                break

            # current frame timestamp in ms
            current_frame_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)

            if current_frame_time_ms >= current_time_ms:
                # save JPEG
                saved_frame_count += 1
                output_path = os.path.join(output_dir, f"frame_{saved_frame_count}.jpg")
                cv2.imwrite(output_path, frame)
                current_time_ms += frame_interval_ms

        video.release()

        print(f"Shranjenih {saved_frame_count} frameov v {output_dir}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    video_path = "assets/video1.mov"  # trenutno imam v assets
    output_dir = "results/video1/"  # hranim v results

    extract_frames(video_path, output_dir, frames_per_second=3)