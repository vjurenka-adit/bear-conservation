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


def to_yolov8_txt_format(bbox, size) -> str:
    """Given a bbox, returns a string line described in the documentation: https://roboflow.com/formats/yolov8-pytorch-txt

    eg. `0 0.617 0.3594420600858369 0.114 0.17381974248927037`
    """
    class_num = 0  # We only detect bear faces
    yolov8 = to_yolov8_bbox(bbox, size)
    return f"{class_num} {yolov8['center_x']} {yolov8['center_y']} {yolov8['w']} {yolov8['h']}"


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
            [to_yolov8_txt_format(bbox, image_size) for bbox in bboxes]
        )
        with open(output_dir / "labels" / f"{filepath.stem}.txt", "w") as f:
            f.write(label_content)
