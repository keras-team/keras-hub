from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.utils.coco_id_to_name")
def coco_id_to_name(id):
    """Convert a single COCO class name to a class ID.

    Args:
        id: An integer class id from 0 to 91.

    Returns:
        The human readable image class name, e.g. "bicycle".

    Example:
    >>> keras_hub.utils.coco_id_to_name(2)
    'bicycle'
    """
    return COCO_NAMES[id]


@keras_hub_export("keras_hub.utils.coco_name_to_id")
def coco_name_to_id(name):
    """Convert a single COCO class name to a class ID.

    Args:
        name: A human readable image class name, e.g. "bicycle".

    Returns:
        The integer class id from 0 to 999.

    Example:
    >>> keras_hub.utils.coco_name_to_id("bicycle")
    2
    """
    return COCO_IDS[name]


COCO_NAMES = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic_light",
    11: "fire_hydrant",
    12: "street_sign",
    13: "stop_sign",
    14: "parking_meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye_glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports_ball",
    38: "kite",
    39: "baseball_bat",
    40: "baseball_glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis_racket",
    44: "bottle",
    45: "plate",
    46: "wine_glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot_dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted_plant",
    65: "bed",
    66: "mirror",
    67: "dining_table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell_phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy_bear",
    89: "hair_drier",
    90: "toothbrush",
    91: "hair_brush",
}

COCO_IDS = {v: k for k, v in COCO_NAMES.items()}
