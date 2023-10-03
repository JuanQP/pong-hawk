import cv2
import numpy as np
import os
import torch

from utilities import (
    MODEL_PATH,
    PROCESSED_FILE_SUFFIX,
    RAINBOW_COLORS,
    VIDEOS_FOLDER,
    center,
    debug_draw,
    process_detection,
)

model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
model.conf = 0.4

# Set this variable with the video name in your videos folder!
files_in_videos_folder = os.listdir(VIDEOS_FOLDER)
print("Files in videos folder:")
print(files_in_videos_folder)
video_filename = input("Video file name (example 'test1.mp4'): ")
draw_debug = input("Export video with detections and heatmaps? (Y/n)").lower()
draw_debug = draw_debug == "y" or draw_debug == ""

if video_filename not in files_in_videos_folder:
    raise RuntimeError(f"File '{video_filename}' does not exist in videos folder")

video = cv2.VideoCapture(f"{VIDEOS_FOLDER}/{video_filename}")
width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(video.get(cv2.CAP_PROP_FPS), 1)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

SCREEN_CENTER = center((0, 0), (width, height))

exported_video = cv2.VideoWriter(
    f"{VIDEOS_FOLDER}/{PROCESSED_FILE_SUFFIX}-{video_filename}",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

debug_exported_video = None
heatmap_exported_video = None

if draw_debug:
    debug_exported_video = cv2.VideoWriter(
        f"{VIDEOS_FOLDER}/{PROCESSED_FILE_SUFFIX}-debug-{video_filename}",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    heatmap_exported_video = cv2.VideoWriter(
        f"{VIDEOS_FOLDER}/{PROCESSED_FILE_SUFFIX}-heatmap-{video_filename}",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

print("")
print("Pong Hawk - Group 33 - Video Processing")
print(f"🛠️  Processing {video_filename} ({width}x{height}@{fps}fps)")

processed_frames = 0
previous_ball_positions = []
# Default values...
last_table_position = SCREEN_CENTER
is_upper_player_playing = None

# Heatmap matrix
# This is where we are going to store
# the ball positions in a grid format
shape = (width // 20, height // 20)
heatmap_matrix = np.zeros(shape, dtype=np.uint8)

# Start processing
while True:
    continues, frame = video.read()

    if not continues:
        break

    previous_frame = frame.copy()

    result = model(frame)

    processed_detections = process_detection(
        result, last_table_position, is_upper_player_playing
    )

    if draw_debug:
        # Draw everything on frame
        debug_frame = frame.copy()
        heatmap_debug_frame = frame.copy()
        debug_draw(processed_detections, debug_frame)

    table = processed_detections["table"]
    if table is not None:
        last_table_position = center(table["start"], table["end"])

    ball = processed_detections["closest_ball"]
    if ball is not None:
        previous_ball_positions.append(ball)
        is_upper_player_playing = (
            center(ball["start"], ball["end"])[1] < last_table_position[1]
        )
        # Update heatmap matrix
        ball_position = center(ball["start"], ball["end"])
        x, y = round(ball_position[0] / 20), round(ball_position[1] / 20)
        heatmap_matrix[x][y] += 1

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
        if draw_debug:
            cv2.circle(
                debug_frame,
                center(ball["start"], ball["end"]),
                radius=2 + extra_radius,
                color=dot_color,
                thickness=-1,
            )

    # Export the image!
    exported_video.write(frame)
    if draw_debug:
        # Map histogram to OpenCV 2D grayscale image and resize it
        # to be the same size as video frame
        max_so_far = heatmap_matrix.max()

        heatmap_grayscale = (
            np.array(heatmap_matrix.T / max_so_far * 255, dtype=np.uint8)
            if max_so_far > 0
            else np.zeros((1, 1), dtype=np.uint8)
        )
        resized = cv2.resize(
            heatmap_grayscale, (width, height), interpolation=cv2.INTER_AREA
        )
        gaussian = cv2.GaussianBlur(resized, (15, 15), 0)
        color_heatmap = cv2.applyColorMap(gaussian, cv2.COLORMAP_JET)

        # "Paste" heatmap in last frame
        combined = cv2.addWeighted(heatmap_debug_frame, 0.5, color_heatmap, 0.5, 0)

        heatmap_exported_video.write(combined)
        # And write to debug video
        debug_exported_video.write(debug_frame)

    processed_frames += 1

    if processed_frames % fps == 0:
        print("", end="\r")
        print(
            f"⏳ {round((processed_frames / frame_count)*100)}% processed",
            end="",
            flush=True,
        )

print("", end="\r")
print(f"✅ 100% processed")
video.release()
exported_video.release()

if draw_debug:
    debug_exported_video.release()
    heatmap_exported_video.release()

# Map histogram to OpenCV 2D grayscale image and resize it
# to be the same size as video frame
heatmap_grayscale = np.array(
    heatmap_matrix.T / heatmap_matrix.max() * 255, dtype=np.uint8
)
resized_heatmap = cv2.resize(
    heatmap_grayscale, (width, height), interpolation=cv2.INTER_AREA
)
blured_heatmap = cv2.GaussianBlur(resized_heatmap, (15, 15), 0)
color_heatmap = cv2.applyColorMap(blured_heatmap, cv2.COLORMAP_JET)

# "Paste" heatmap in last frame
combined = cv2.addWeighted(previous_frame, 0.5, color_heatmap, 0.5, 0)
heatmap_filename = f"{video_filename.split('.')[0]}.png"
cv2.imwrite(
    f"{VIDEOS_FOLDER}/{PROCESSED_FILE_SUFFIX}-heatmap-{heatmap_filename}", combined
)

cv2.destroyAllWindows()
print("Done!")
