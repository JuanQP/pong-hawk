import cv2
import torch

from utilities import (COLORS, MAX_AMOUNTS, MODEL_PATH, PROCESSED_FILE_SUFFIX,
                       VIDEOS_FOLDER, image_to_rgb)
from utilities.videos import (ACTIVE_PLAYER_COLORS, RAINBOW_COLORS, center,
                              distance, draw_text)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.conf = 0.4

# Set this variable with the video name in your videos folder!
VIDEO_FILENAME = "some_filename_in_videos_folder.mp4"

video = cv2.VideoCapture(f"{VIDEOS_FOLDER}/{VIDEO_FILENAME}")
width, height, fps = 1280, 720, 25.0
SCREEN_CENTER = center((0,0), (width, height))
last_table_position = SCREEN_CENTER

exported_video = cv2.VideoWriter(
    f"{VIDEOS_FOLDER}/{PROCESSED_FILE_SUFFIX}{VIDEO_FILENAME}",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

processed_frames = 0
previous_ball_positions = []
# Default value...
is_upper_player_playing = True
while True:
    continues, frame = video.read()

    if not continues:
        break

    rgb_frame = image_to_rgb(frame)
    result = model(rgb_frame)

    detected_objects = result.pandas().xyxy[0]
    object_count = {object_name: 0 for object_name in MAX_AMOUNTS.keys()}
    ball_rectangle_positions = []
    player_positions = []

    for i in detected_objects.index:
        object_name = detected_objects['name'][i]
        object_count[object_name] += 1
        max_count_exceeded = object_count[object_name] > MAX_AMOUNTS[object_name]

        if max_count_exceeded and object_name != 'pelota':
            continue

        confidence = detected_objects['confidence'][i]
        start_point = int(detected_objects['xmin'][i]), int(detected_objects['ymin'][i])
        end_point = int(detected_objects['xmax'][i]), int(detected_objects['ymax'][i])
        label = f"{object_name} {round(confidence * 100)}%"

        if object_name == 'jugador':
            player_positions.append((start_point, end_point, label))
            continue
        elif object_name == 'mesa':
            last_table_position = center(start_point, end_point)
        elif object_name == 'pelota':
            ball_rectangle_positions.append((start_point, end_point))
            continue


        colors = COLORS[object_name]
        border_color = colors['background']

        # Rectangle
        cv2.rectangle(frame, start_point, end_point, border_color, 1)
        draw_text(label, start_point, colors, frame)

    closest_ball = None
    if len(ball_rectangle_positions) > 0:
        # Find the closest ball to the table
        closest_ball = min(
            ball_rectangle_positions,
            key=lambda p: distance(last_table_position, center(p[0], p[1]))
        )
        previous_ball_positions.append(closest_ball)
        cv2.rectangle(frame, closest_ball[0], closest_ball[1], COLORS['pelota']['background'], 1)
        draw_text('pelota', closest_ball[0], COLORS['pelota'], frame)

    extra_radius = 0
    for i, dot in enumerate(previous_ball_positions[-7:]):
        extra_radius = i // 3
        dot_color = RAINBOW_COLORS[i]
        cv2.circle(frame, center(dot[0], dot[1]), radius=2 + extra_radius, color=dot_color, thickness=-1)

    if len(previous_ball_positions) > 0:
        last_ball_position = center(previous_ball_positions[-1][0], previous_ball_positions[-1][1])
        is_upper_player_playing = last_ball_position[1] > last_table_position[1]

    for start, end, label in player_positions:
        player_position = center(start, end)
        is_upper_player = player_position[1] > last_table_position[1]
        player_color = ACTIVE_PLAYER_COLORS \
            if (is_upper_player and is_upper_player_playing) or (not is_upper_player and not is_upper_player_playing) \
            else COLORS['jugador']
        cv2.rectangle(frame, start, end, player_color['background'], 1)
        draw_text(label, start, player_color, frame)

    processed_frame = frame
    exported_video.write(processed_frame)
    processed_frames += 1

    if processed_frames % fps == 0:
        print(f"{processed_frames / fps} seconds processed...")

video.release()
exported_video.release()
cv2.destroyAllWindows()
