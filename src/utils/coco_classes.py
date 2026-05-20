"""COCO 80 class names (YOLO / Ultralytics order)."""

from __future__ import annotations

# Standard COCO 2017 order used by Ultralytics YOLO
COCO80_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO80_IDS: tuple[int, ...] = tuple(range(80))

# CCTV preset: person + common road users / vehicles
CCTV_TRAFFIC_CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


def id_to_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO80_NAMES):
        return COCO80_NAMES[class_id]
    return f"class_{class_id}"


def name_to_id(name: str) -> int | None:
    n = name.strip().lower()
    for i, cn in enumerate(COCO80_NAMES):
        if cn == n:
            return i
    return None


def coco_name_dict() -> dict[int, str]:
    return {i: COCO80_NAMES[i] for i in range(80)}
