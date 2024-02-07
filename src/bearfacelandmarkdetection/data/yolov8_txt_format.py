import os
import shutil
from pathlib import Path

from tqdm import tqdm


def to_yolov8_bbox(bbox, size) -> dict:
    """Returns a dict containing the parameters outlined in the yolov8 txt format doc: https://roboflow.com/formats/yolov8-pytorch-txt

    - center_x: float between 0 and 1
    - center_y: float between 0 and 1
    - w: float between 0 and 1
    - h: float between 0 and 1
    """
    center_x = (bbox["left"] + bbox["width"] / 2.0) / size["width"]
    center_y = (bbox["top"] + bbox["height"] / 2.0) / size["height"]
    w = bbox["width"] / size["width"]
    h = bbox["height"] / size["height"]

    assert 0.0 <= center_x <= 1.0, "center_x should be between 0 and 1"
    assert 0.0 <= center_y <= 1.0, "center_y should be between 0 and 1"
    assert 0.0 <= w <= 1.0, "w should be between 0 and 1"
    assert 0.0 <= h <= 1.0, "h should be between 0 and 1"

    return {"center_x": center_x, "center_y": center_y, "w": w, "h": h}


def to_yolov8_point(x, y, size) -> dict:
    """Returns a dict containing the parameters outlined in the yolov8 txt format doc: https://roboflow.com/formats/yolov8-pytorch-txt

    - x: float between 0 and 1
    - y: float between 0 and 1
    """
    normalized_x = x / size["width"]
    normalized_y = y / size["height"]

    assert 0.0 <= normalized_x <= 1.0, "normalized_x should be between 0 and 1"
    assert 0.0 <= normalized_y <= 1.0, "normalized_y should be between 0 and 1"

    return {"x": normalized_x, "y": normalized_y}


def to_yolov8_txt_format(
    bbox,
    nose_point: tuple[int, int],
    leye_point: tuple[int, int],
    reye_point: tuple[int, int],
    size: dict,
    keypoints_order: list = ["nose", "leye", "reye"],
) -> str:
    """
    Following the doc from here: https://docs.ultralytics.com/datasets/pose/#ultralytics-yolo-format
    """
    class_index = 0
    normalized_bbox = to_yolov8_bbox(bbox, size)
    normalized_point_nose = to_yolov8_point(nose_point[0], nose_point[1], size)
    normalized_point_leye = to_yolov8_point(leye_point[0], leye_point[1], size)
    normalized_point_reye = to_yolov8_point(reye_point[0], reye_point[1], size)
    keypoints = {
        "nose": normalized_point_nose,
        "leye": normalized_point_leye,
        "reye": normalized_point_reye,
    }
    ordered_keypoints = [keypoints[label] for label in keypoints_order]
    keypoints_str = " ".join([f"{kp['x']} {kp['y']}" for kp in ordered_keypoints])
    return f"{class_index} {normalized_bbox['center_x']} {normalized_bbox['center_y']} {normalized_bbox['w']} {normalized_bbox['h']} {keypoints_str}"


def part_to_point(part):
    return part["x"], part["y"]


def build_yolov8_txt_format(xml_data, output_dir: Path) -> None:
    """Given the xml_data parsed from bearID, it generates the bare yolov8 txt
    format and save it in `output_dir`"""
    # Creating the directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "images", exist_ok=True)
    os.makedirs(output_dir / "labels", exist_ok=True)

    for image_data in tqdm(xml_data["images"]):
        filepath = image_data["filepath"]
        bboxes = image_data["bboxes"]
        image_size = image_data["size"]

        # Copying the images
        shutil.copy(filepath, output_dir / "images" / filepath.name)

        # Making the label files
        label_content = "\n".join(
            [
                to_yolov8_txt_format(
                    bbox=bbox,
                    nose_point=part_to_point(bbox["parts"]["nose"]),
                    leye_point=part_to_point(bbox["parts"]["leye"]),
                    reye_point=part_to_point(bbox["parts"]["reye"]),
                    size=image_size,
                )
                for bbox in bboxes
            ]
        )
        with open(output_dir / "labels" / f"{filepath.stem}.txt", "w") as f:
            f.write(label_content)
