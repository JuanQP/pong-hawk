import os

import cv2
import torch

from utilities import (COLORS, IMAGES_FOLDER, MAX_AMOUNTS, MODEL_PATH,
                       PROCESSED_FILE_SUFFIX, image_to_rgb)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.conf = 0.4

image_files = os.listdir(IMAGES_FOLDER)
print("Pong Hawk - Group 33 - Image Processing")
print("The next files will be processed:", image_files)
for image_path in image_files:
    if image_path.startswith(PROCESSED_FILE_SUFFIX):
        print(f"Looks like {image_path} is already a processed image... Skipping...")
    object_count = { object_name: 0 for object_name in MAX_AMOUNTS.keys() }
    image = cv2.imread(f"{IMAGES_FOLDER}/{image_path}")
    if image is None:
        print(f"Looks like {image_path} is not an image... Skipping...")
        continue
    image_rgb = image_to_rgb(image)
    result = model(image_rgb)
    detections = result.pandas().xyxy[0]

    # Iterate over the detections in current image
    for i in detections.index:
        object_name = detections['name'][i]
        object_count[object_name] += 1
        is_greater_than_max_count = object_count[object_name] > MAX_AMOUNTS[object_name]

        confidence = detections['confidence'][i]
        start_point = int(detections['xmin'][i]), int(detections['ymin'][i])
        end_point = int(detections['xmax'][i]), int(detections['ymax'][i])
        label_position = (start_point[0], start_point[1] - 10)
        label = f"{object_name} {round(confidence * 100)}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_background_end_position = (start_point[0] + w, start_point[1] - h * 2)
        colors = COLORS[object_name]
        border_color = (0, 0, 255) if is_greater_than_max_count else colors['background']

        # Rectangle
        cv2.rectangle(image, start_point, end_point, border_color, 2)
        # Label
        cv2.rectangle(image, start_point, label_background_end_position, border_color, -1)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['text'], 2)

    # Export result image
    processed_image_filename = f"{PROCESSED_FILE_SUFFIX}{image_path}"
    cv2.imwrite(f"{IMAGES_FOLDER}/{processed_image_filename}", image)
    print(f"Exported {image_path} -> {processed_image_filename}")

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!")
