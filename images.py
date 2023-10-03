import os

import cv2
import numpy as np
import torch

from utilities import (
    IMAGES_FOLDER,
    MODEL_PATH,
    PROCESSED_FILE_SUFFIX,
    center,
    debug_draw,
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

alpha = 0.2

for image_path in image_files:
    if image_path.startswith(PROCESSED_FILE_SUFFIX):
        print(f"Looks like {image_path} is already a processed image... Skipping...")
        continue

    image = cv2.imread(f"{IMAGES_FOLDER}/{image_path}")
    if image is None:
        print(f"Looks like {image_path} is not an image... Skipping...")
        continue

    result = model(image)
    image_center = center((0, 0), image.shape[:2])

    processed_detections = process_detection(result, image_center)

    # Draw everything on image
    debug_draw(processed_detections, image)

    # Export image with all detections drawn on it
    processed_image_filename = f"{PROCESSED_FILE_SUFFIX}{image_path}"
    cv2.imwrite(f"{IMAGES_FOLDER}/{processed_image_filename}", image)
    print(f"Exported {image_path} -> {processed_image_filename}")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!")
