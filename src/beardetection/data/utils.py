import json
import logging
from os import path
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm

import bearfacelabeling.predict


def bbox_to_string(bbox: list[float]) -> str:
    x, y, width, height = bbox
    return f"{x} {y} {width} {height}"


def parse_bbox(s: str) -> list[float]:
    x, y, width, height = [float(x) for x in s.split(" ")]
    return [x, y, width, height]


def get_best_device() -> torch.device:
    """Returns the best torch device depending on the hardware it is running
    on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_groundingDINO_model(device: str, model_checkpoint_path: Path):
    return bearfacelabeling.predict.Model(
        model_checkpoint_path,
        device=device,
    )


def annotate(
    model,
    text_prompt: str,
    token_spans: list,
    image_paths: list[str],
    input_dir: Path,
    output_dir: Path,
) -> None:
    for image_path in tqdm(image_paths):
        try:
            image = Image.open(image_path)
            bbox = model.predict(
                image,
                text_prompt=text_prompt,
                token_spans=token_spans,
            )

            relative_image_path = path.relpath(image_path, input_dir)
            relative_ann_path = path.splitext(relative_image_path)[0] + ".txt"
            ann_path = path.join(output_dir, relative_ann_path)

            # Create dir
            Path(path.dirname(ann_path)).mkdir(parents=True, exist_ok=True)

            with open(ann_path, "w") as f:
                if bbox:
                    f.write(bbox_to_string(bbox))
        except:
            logging.warning(f"image {image_path} cannot be read or processed")


def parse_annotations(
    input_dir: Path,
    output_dir: Path,
    image_paths: list[Path],
) -> dict[str, pd.DataFrame]:
    anns = {}
    imgs_without_ann_file = []

    for img_path in image_paths:
        img_rel_path = relative_image_path = path.relpath(img_path, input_dir)
        ann_rel_path = path.splitext(img_rel_path)[0] + ".txt"

        try:
            with open(path.join(output_dir, ann_rel_path), "r") as f_ann:
                content = f_ann.read()
                if not content:
                    anns[img_rel_path] = None
                else:
                    lines = content.split("\n")
                    bboxes = [parse_bbox(line) for line in lines]
                    if bboxes:
                        anns[img_rel_path] = bboxes[0]
                    else:
                        anns[img_rel_path] = None
        except FileNotFoundError as err:
            imgs_without_ann_file.append(img_rel_path)

    df_image_pb = pd.DataFrame.from_dict({"img": imgs_without_ann_file})
    df = pd.DataFrame.from_dict({"img": anns.keys(), "bbox": anns.values()})

    return {"ok": df, "ko": df_image_pb}


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def write_data_yaml(path: Path) -> None:
    """Writes the `data.yaml` file necessary for YOLOv8 training at `path`
    location."""
    data = {
        "train": "./train/images",
        "val": "./val/images",
        "test": "./test/images",
        "nc": 1,
        "names": ["bear"],
    }
    with open(path / "data.yaml", "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def get_annotation_filepaths(input_dir: Path) -> list[Path]:
    return list(input_dir.rglob("*.txt"))


def is_valid_annotation_filepath(filepath: Path) -> bool:
    """Checks whether the annotation filepath is valid."""
    if not filepath.exists():
        logging.warning(f"invalid annotation in {filepath} - does not exist")
        return False
    else:
        try:
            with open(filepath, "r") as f:
                content = f.readline()
                splitted = content.split(" ")
                if len(splitted) == 5:
                    return True
                else:
                    logging.warning(
                        f"invalid annotation in {filepath} - does not have 5 floats"
                    )
                    return False
        except:
            logging.warning(f"invalid annotation in {filepath} - can't open file")
            return False


def label_filepath_to_image_filepath(
    input_dir_hack_the_planet: Path,
    input_dir: Path,
    label_filepath: Path,
) -> Optional[Path]:
    relative_label_filepath = label_filepath.relative_to(input_dir)
    possible_image_filepaths = []
    for extension in ["jpg", "PNG", "JPG"]:
        image_filepath = (
            input_dir_hack_the_planet
            / relative_label_filepath.parent
            / f"{relative_label_filepath.stem}.{extension}"
        )
        possible_image_filepaths.append(image_filepath)

    for possible_image_filepath in possible_image_filepaths:
        if possible_image_filepath.exists():
            return possible_image_filepath

    return None


def exif(image_filepath: Path) -> dict:
    """Returns exif data from the image."""
    if not image_filepath.exists():
        logging.error(f"image_filepath {image_filepath} not found")
        return {}
    else:
        try:
            image = Image.open(image_filepath)
            return {TAGS[k]: v for k, v in image.getexif().items() if k in TAGS}
        except:
            logging.error(f"Can't extract exif data from {image_filepath}")
            return {}


def slurp_json(json_filepath: Path):
    """Parsers the json filepath and returns its content."""
    with open(json_filepath, "r") as f:
        return json.load(f)


def is_valid_image_filepath(image_filepath: Path) -> bool:
    """Checks whether the image_filepath exists and can be read."""
    if not image_filepath.exists():
        return False
    else:
        try:
            Image.open(image_filepath)
            return True
        except:
            return False


def get_image_filepaths_without_bears(input_dir_hack_the_planet: Path) -> list[Path]:
    """Returns a list of image filepaths that do not contain bears."""
    images_root_dir = input_dir_hack_the_planet / "images"
    coco_filepath = input_dir_hack_the_planet / "coco.json"

    assert images_root_dir.exists()
    assert coco_filepath.exists()

    logging.info(f"Parsing coco_filepath: {coco_filepath}")
    coco_data = slurp_json(coco_filepath)
    if not coco_data:
        logging.error(f"could not load coco_filepath: {coco_filepath}")
        return []
    else:
        id_to_image_data = {
            image_data["id"]: image_data for image_data in coco_data["images"]
        }
        label_to_class_id = {
            category["name"]: category["id"] for category in coco_data["categories"]
        }
        annotations = coco_data["annotations"]
        annotations_without_bears = [
            annotation
            for annotation in annotations
            if (annotation["category_id"] != label_to_class_id["bear"])
        ]
        image_filepaths_without_bears = [
            images_root_dir / id_to_image_data[annotation["image_id"]]["file_name"]
            for annotation in annotations_without_bears
        ]
        return [
            image_filepath
            for image_filepath in image_filepaths_without_bears
            if is_valid_image_filepath(image_filepath)
        ]


def load_datasplit(filepath: Path) -> pd.DataFrame:
    """Loads the datasplit."""
    df_split = pd.read_csv(filepath, sep=";")
    df_split = df_split.replace({np.nan: None})
    return df_split


## REPL
# image_filepath = Path(
#     "data/01_raw/Hack the Planet/images/Season2 - bears only/22RucarAG/149_Mara mare/20190320_20190412/Bushn_CC00Y8/03200029.JPG"
# )

# data = exif(image_filepath)

# data["DateTime"]

# from dateutil import parser

# t = parser.parse(timestr=data["DateTime"])
# t.year
# t.month
# t.day
# t
