import logging
import os
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from bearfacedetection.xml_parser import load_xml


def load_sam_predictor(sam_checkpoint_path: Path):
    """Loads the SAM model using the `sam_checkpoint_path` and returns a
    SamPredictor."""
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    return SamPredictor(sam)


def load_image(image_data: dict) -> np.ndarray:
    image_filepath = image_data["filepath"]
    image = cv2.imread(str(image_filepath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def select_best(masks, scores):
    best_idx = np.argmax(scores)
    return masks[best_idx], scores[best_idx]


def run(
    predictor: SamPredictor,
    base_path: Path,
    label_path: Path,
    output_dir: Path,
) -> None:
    """Main function to run the sam predictor and generate segmentation masks
    for bears.

    `base_path`: bearid base path
    `label_path`: xml filepath that contains the annotations from bearid
    `output_dir`: path to save the results
    """
    logging.info(f"Parsing the xml dataset from {label_path}")
    xml_data = load_xml(base_path, label_path)

    if not os.path.isdir(output_dir):
        logging.info(f"Making directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    logging.info("Segmenting the images with SAM HQ")
    for image_data in tqdm(xml_data["images"]):
        stem = image_data["filepath"].stem
        # We assume that there is only one bbox
        parts = image_data["bboxes"][0]["parts"]
        image = load_image(image_data)
        predictor.set_image(image)
        input_point = np.array(
            [
                [parts["nose"]["x"], parts["nose"]["y"]],
                [parts["leye"]["x"], parts["leye"]["y"]],
                [parts["reye"]["x"], parts["reye"]["y"]],
            ]
        )
        input_label = np.array([1, 1, 1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask, _ = select_best(masks, scores)
        out_filepath = output_dir / f"{stem}.png"
        cv2.imwrite(str(out_filepath), best_mask * 255)
