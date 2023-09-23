import math

import cv2

ACTIVE_PLAYER_COLORS = {
    'text': (0, 0, 0),
    'background': (0, 215, 255),
}

RAINBOW_COLORS = [
    (255, 0, 127),
    (106, 65, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 127, 255),
    (0, 0, 255),
]

def center(start, end):
    return (
        int((start[0] + end[0]) / 2),
        int((start[1] + end[1]) / 2),
    )

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def draw_text(text: str, start_position, colors, frame):
    label_position = (start_position[0], start_position[1] - 10)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    label_background_end_position = (start_position[0] + w, start_position[1] - h * 2)

    # Label
    cv2.rectangle(frame, start_position, label_background_end_position, colors['background'], -1)
    cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['text'], 2)
