import os

import cv2
import torch

from utilities import (
    IMAGES_FOLDER,
    MAX_AMOUNTS,
    MODEL_PATH,
    PROCESSED_FILE_SUFFIX,
    draw_detection,
    image_to_rgb,
    to_detection,
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

    # Initialize detections in current image
    detections_by_type = {object_name: 0 for object_name in MAX_AMOUNTS.keys()}
    image_rgb = image_to_rgb(image)
    result = model(image_rgb)
    detections = result.pandas().xyxy[0]

    # Iterate over the detections in current image
    for i in detections.index:
        detection = to_detection(detections.iloc[i])
        detection_name = detection["name"]
        detections_by_type[detection_name] += 1
        draw_detection(image, detections_by_type, detection)

    # Export image with all detections drawn on it
    processed_image_filename = f"{PROCESSED_FILE_SUFFIX}{image_path}"
    cv2.imwrite(f"{IMAGES_FOLDER}/{processed_image_filename}", image)
    print(f"Exported {image_path} -> {processed_image_filename}")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!")
