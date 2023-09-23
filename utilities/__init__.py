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
