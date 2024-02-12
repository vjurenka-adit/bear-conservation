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


def run(
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
