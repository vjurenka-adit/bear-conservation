import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm


def get_best_device() -> torch.device:
    """Returns the best torch device depending on the hardware it is running
    on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def yaml_read(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
            for annotation in tqdm(annotations)
            if (annotation["category_id"] != label_to_class_id["bear"])
        ]
        logging.info(
            f"loading annotations without bears {len(annotations_without_bears)}"
        )
        image_filepaths_without_bears = [
            images_root_dir / id_to_image_data[annotation["image_id"]]["file_name"]
            for annotation in annotations_without_bears
        ]
        logging.info(
            f"loading image filepaths without bears - {len(image_filepaths_without_bears)}"
        )
        result = [
            image_filepath
            for image_filepath in tqdm(image_filepaths_without_bears)
            if is_valid_image_filepath(image_filepath)
        ]
        logging.info(f"filtering out invalid image filepaths. - {len(result)}")
        return result


def load_datasplit(filepath: Path) -> pd.DataFrame:
    """Loads the datasplit."""
    df_split = pd.read_csv(filepath, sep=";")
    df_split = df_split.replace({np.nan: None})
    return df_split


def downsample_by_group(
    df: pd.DataFrame,
    ratio: float,
    groupby_key: list = ["camera_id", "year", "month", "day"],
) -> pd.DataFrame:
    """Returns a pd.Index corresponding to the indices to keep to perform some
    downsampling."""
    df_result = df.copy().reset_index()
    g = df_result.groupby(groupby_key)
    G = list(g.groups.items())
    downsampled_indices = []
    for _, indices in G:
        k = int(max(1, round(number=ratio * len(indices), ndigits=0)))
        xs = random.sample(list(indices), k=k)
        downsampled_indices.extend(xs)
    return df_result.iloc[downsampled_indices]


def get_ratio_others_over_bears(df_split: pd.DataFrame) -> float:
    df_counts = df_split.copy().groupby("class").count().reset_index("class")
    n_bears = df_counts[df_counts["class"] == "bear"]["image_filepath"].iloc[0]
    n_others = df_counts[df_counts["class"] == "other"]["image_filepath"].iloc[0]
    return n_others / n_bears


def balance_classes(df_split: pd.DataFrame) -> pd.DataFrame:
    """Given a df_split dataframe, it rebalances the classes at the group level
    (burst of images from the same camera) using downsampling techniques.

    It can allow a model to train faster and better.
    """
    ratio = get_ratio_others_over_bears(df_split=df_split)
    df_others = df_split[df_split["class"] == "other"]
    df_bears = df_split[df_split["class"] == "bear"]
    if ratio <= 1:
        df_bears_downsampled = downsample_by_group(df=df_bears, ratio=ratio)
        df_split_downsampled = pd.concat(
            [df_bears_downsampled, df_others.copy().reset_index()]
        )

        return df_split_downsampled
    else:
        df_others_downsampled = downsample_by_group(df=df_others, ratio=ratio)
        df_split_downsampled = pd.concat(
            [df_bears.copy().reset_index(), df_others_downsampled]
        )
        return df_split_downsampled


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
