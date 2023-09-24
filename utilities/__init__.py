import cv2

RED_COLOR = (0, 0, 255)

COLORS = {
    'pelota': {
        'text': (0, 0, 0),
        'background': (255, 255, 255),
    },
    'jugador': {
        'text': (255, 255, 255),
        'background': (225, 105, 65),
    },
    'paleta': {
        'text': (255, 255, 255),
        'background': (220,20,60),
    },
    'red': {
        'text': (0, 0, 0),
        'background': (178,190,181),
    },
    'mesa': {
        'text': (255, 255, 255),
        'background': (34,139,34),
    },
}
IMAGES_FOLDER = "images"
MAX_AMOUNTS = {
    'pelota': 1,
    'jugador': 2,
    'paleta': 2,
    'mesa': 1,
    'red': 1,
}
MODEL_PATH = "model/trained_model.pt"
PROCESSED_FILE_SUFFIX = "pong-hawk-"
VIDEOS_FOLDER = "videos"

def image_to_rgb(image):
    return image[..., ::-1]

def to_detection(row):
    detection_name = row['name']

    return {
        "name": detection_name,
        "confidence": row['confidence'],
        "start": (int(row['xmin']), int(row['ymin'])),
        "end": (int(row['xmax']), int(row['ymax'])),
        "colors": COLORS[detection_name],
    }

def draw_detection(image, detections_by_type, detection):
    start_point = detection["start"]
    detection_name = detection["name"]
    confidence = detection["confidence"]
    colors = detection["colors"]
    is_greater_than_max_count = detections_by_type[detection_name] > MAX_AMOUNTS[detection_name]

    # Some positioning for the label and background...
    label_position = (start_point[0], start_point[1] - 10)
    label_text = f"{detection_name} {round(confidence * 100)}%"
    (width, height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    label_background_end_position = (start_point[0] + width, start_point[1] - height * 2)
    border_color = RED_COLOR if is_greater_than_max_count else colors['background']

    # Draw a rectangle
    cv2.rectangle(image, detection["start"], detection["end"], border_color, thickness=2)
    # Draw text in image with background
    cv2.rectangle(image, detection["start"], label_background_end_position, border_color, thickness=-1)
    cv2.putText(image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, detection['colors']['text'], 2)
