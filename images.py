import os

import cv2
import torch

from utilities import (
    IMAGES_FOLDER,
    MODEL_PATH,
    PROCESSED_FILE_SUFFIX,
    center,
    draw_detection,
    image_to_rgb,
    process_detection,
)

# Load model
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)
model.conf = 0.4
# Get files in images folder
image_files = os.listdir(IMAGES_FOLDER)
print("")
print("Pong Hawk - Group 33 - Image Processing")
print("The next files will be processed:", image_files)

for image_path in image_files:
    if image_path.startswith(PROCESSED_FILE_SUFFIX):
        print(f"Looks like {image_path} is already a processed image... Skipping...")
        continue

    image = cv2.imread(f"{IMAGES_FOLDER}/{image_path}")
    if image is None:
        print(f"Looks like {image_path} is not an image... Skipping...")
        continue

    rgb_frame = image_to_rgb(image)
    result = model(rgb_frame)
    image_center = center((0, 0), image.shape[:2])

    processed_detections = process_detection(result, image_center)

    table = processed_detections["table"]
    if table is not None:
        draw_detection(image, table)

    web = processed_detections["web"]
    if web is not None:
        draw_detection(image, web)

    for paddle in processed_detections["paddles"]:
        draw_detection(image, paddle)

    for player in processed_detections["players"]:
        draw_detection(image, player)

    ball = processed_detections["closest_ball"]
    if ball is not None:
        draw_detection(image, ball)

    # Draw boundaries
    left_boundary = processed_detections["boundaries"][0]
    right_boundary = processed_detections["boundaries"][1]
    image_height = image.shape[1]

    cv2.line(
        image,
        (left_boundary, 0),
        (left_boundary, image_height),
        color=(0, 0, 0),
        thickness=1,
    )
    cv2.line(
        image,
        (right_boundary, 0),
        (right_boundary, image_height),
        color=(0, 0, 0),
        thickness=1,
    )

    # Export image with all detections drawn on it
    processed_image_filename = f"{PROCESSED_FILE_SUFFIX}{image_path}"
    cv2.imwrite(f"{IMAGES_FOLDER}/{processed_image_filename}", image)
    print(f"Exported {image_path} -> {processed_image_filename}")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!")
