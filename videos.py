import cv2
import os
import torch

from utilities import (
    MODEL_PATH,
    PROCESSED_FILE_SUFFIX,
    RAINBOW_COLORS,
    VIDEOS_FOLDER,
    center,
    draw_detection,
    image_to_rgb,
    process_detection,
)

model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
model.conf = 0.4

# Set this variable with the video name in your videos folder!
files_in_videos_folder = os.listdir(VIDEOS_FOLDER)
print("Files in videos folder:")
print(files_in_videos_folder)
video_filename = input("Video file name (example 'test1.mp4'): ")

if video_filename not in files_in_videos_folder:
    raise RuntimeError(f"File '{video_filename}' does not exist in videos folder")

video = cv2.VideoCapture(f"{VIDEOS_FOLDER}/{video_filename}")
width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(video.get(cv2.CAP_PROP_FPS), 1)

SCREEN_CENTER = center((0, 0), (width, height))

exported_video = cv2.VideoWriter(
    f"{VIDEOS_FOLDER}/{PROCESSED_FILE_SUFFIX}{video_filename}",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

print("")
print("Pong Hawk - Group 33 - Video Processing")
print(f"Processing {video_filename} ({width}x{height}@{fps}fps)")

processed_frames = 0
previous_ball_positions = []
# Default values...
last_table_position = SCREEN_CENTER
is_upper_player_playing = None

# Start processing
while True:
    continues, frame = video.read()

    if not continues:
        break

    rgb_frame = image_to_rgb(frame)
    result = model(rgb_frame)

    processed_detections = process_detection(
        result, last_table_position, is_upper_player_playing
    )

    table = processed_detections["table"]
    if table is not None:
        last_table_position = center(table["start"], table["end"])
        draw_detection(frame, table)

    web = processed_detections["web"]
    if web is not None:
        draw_detection(frame, web)

    for paddle in processed_detections["paddles"]:
        draw_detection(frame, paddle)

    for player in processed_detections["players"]:
        draw_detection(frame, player)

    ball = processed_detections["closest_ball"]
    if ball is not None:
        previous_ball_positions.append(ball)
        is_upper_player_playing = (
            center(ball["start"], ball["end"])[1] < last_table_position[1]
        )
        draw_detection(frame, ball)

    # Draw a rainbow trail
    extra_radius = 0
    for i, ball in enumerate(previous_ball_positions[-7:]):
        # Make more recent dots larger
        extra_radius = i // 3
        dot_color = RAINBOW_COLORS[i]
        cv2.circle(
            frame,
            center(ball["start"], ball["end"]),
            radius=2 + extra_radius,
            color=dot_color,
            thickness=-1,
        )

    exported_video.write(frame)
    processed_frames += 1

    if processed_frames % fps == 0:
        print(f"{processed_frames / fps} seconds processed...")

video.release()
exported_video.release()
cv2.destroyAllWindows()
