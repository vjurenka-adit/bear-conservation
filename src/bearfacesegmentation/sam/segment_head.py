import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from bearfacedetection.xml_parser import load_xml


def filepath_to_image(filepath: Path):
    image = cv2.imread(str(filepath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def to_boolean_mask(black_white_image):
    return np.all(black_white_image == [255, 255, 255], axis=2)


def load_image(image_data: dict) -> np.ndarray:
    image_filepath = image_data["filepath"]
    image = cv2.imread(str(image_filepath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def run_with_bearid_xml(
    bearid_base_path: Path,
    label_path: Path,
    masks_body_dir: Path,
    output_dir: Path,
) -> None:
    """Main function to run the sam predictor and generate segmentation masks
    for bears.

    `bearid_base_path`: bearid base path
    `label_path`: xml filepath that contains the annotations from bearid
    `output_dir`: path to save the results
    """
    logging.info(f"Parsing the xml dataset from {label_path}")
    xml_data = load_xml(bearid_base_path, label_path)

    if not os.path.isdir(output_dir):
        logging.info(f"Making directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    masks_body_filepaths = [
        masks_body_dir / filename for filename in os.listdir(masks_body_dir)
    ]
    stem_to_mask_filepath = {
        filepath.stem: filepath for filepath in masks_body_filepaths
    }

    logging.info("Combining bboxes and body masks to generate bear heads masks")
    for image_data in tqdm(xml_data["images"]):
        stem = image_data["filepath"].stem
        mask_body_filepath = stem_to_mask_filepath.get(stem)
        if mask_body_filepath:
            boolean_mask_body = to_boolean_mask(filepath_to_image(mask_body_filepath))
            boolean_mask_head = np.zeros(boolean_mask_body.shape, bool)
            # We assume that there is only one bbox per image
            bbox = image_data["bboxes"][0]
            boolean_mask_head[
                bbox["top"] : bbox["top"] + bbox["height"],
                bbox["left"] : bbox["left"] + bbox["width"],
            ] = boolean_mask_body[
                bbox["top"] : bbox["top"] + bbox["height"],
                bbox["left"] : bbox["left"] + bbox["width"],
            ]
            out_filepath = output_dir / f"{stem}.png"
            cv2.imwrite(str(out_filepath), boolean_mask_head * 255)


def to_stem(roboflow_filename: str):
    """Given a roboflow filename, it returns its stem.

    >>> to_stem('P1250760_JPG.rf.f193811e6d745a717b7dc4b80f2219d0.jpg')
    """
    return (
        roboflow_filename.split(".rf")[0]
        .replace("_jpg", "")
        .replace("_JPG", "")
        .replace("_png", "")
        .replace("_PNG", "")
    )


def parse_yolov8_txt_bbox_line(s: str):
    class_str, center_x_str, center_y_str, w_str, h_str = s.split(" ")
    return {
        "class": int(class_str),
        "center_x": float(center_x_str),
        "center_y": float(center_y_str),
        "w": float(w_str),
        "h": float(h_str),
    }


def run_with_yolov8_labels(
    label_path: Path,
    masks_body_dir: Path,
    output_dir: Path,
) -> None:
    """Main function to run the sam predictor and generate segmentation masks
    for bears.

    `bearid_base_path`: bearid base path
    `label_path`: root directory containing the images and labels (export from roboflow in the yolov8 txt format)
    `output_dir`: path to save the results
    """

    if not os.path.isdir(output_dir):
        logging.info(f"Making directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    masks_body_filepaths = [
        masks_body_dir / filename for filename in os.listdir(masks_body_dir)
    ]
    stem_to_mask_filepath = {
        filepath.stem.replace(" ", "-"): filepath for filepath in masks_body_filepaths
    }
    # Fix the error file
    # stem_to_mask_filepath[
    #     "854-Divot-with-snare-near-Big-Creek-from-CalliopeJane-July-24-2014-"
    # ] = (
    #     masks_body_dir
    #     / "854 Divot with snare near Big Creek from CalliopeJane  (July 24, 2014).jpg"
    # )

    logging.info("Combining bboxes and masks to generate bear heads masks")
    for label_filename in tqdm(os.listdir(label_path / "labels")):
        stem = to_stem(label_filename)
        mask_body_filepath = stem_to_mask_filepath.get(stem)
        if not mask_body_filepath:
            logging.error(
                f"cannot find associated mask filepath -- stem: {stem} -- label_filename: {label_filename}"
            )
        else:
            boolean_mask_body = to_boolean_mask(filepath_to_image(mask_body_filepath))
            boolean_mask_head = np.zeros(boolean_mask_body.shape, bool)
            # We assume that there is only one bbox per image
            H, W = boolean_mask_body.shape
            with open(label_path / "labels" / label_filename) as f:
                data = parse_yolov8_txt_bbox_line(f.readline())
                bbox = {
                    "top": int((data["center_y"] - data["h"] / 2) * H),
                    "left": int((data["center_x"] - data["w"] / 2) * W),
                    "width": int(data["w"] * W),
                    "height": int(data["h"] * H),
                }
                boolean_mask_head[
                    bbox["top"] : bbox["top"] + bbox["height"],
                    bbox["left"] : bbox["left"] + bbox["width"],
                ] = boolean_mask_body[
                    bbox["top"] : bbox["top"] + bbox["height"],
                    bbox["left"] : bbox["left"] + bbox["width"],
                ]
                out_filepath = output_dir / f"{Path(label_filename).stem}.png"
                cv2.imwrite(str(out_filepath), boolean_mask_head * 255)
