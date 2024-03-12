import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from tqdm import tqdm


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
        "names": ["bearbody"],
    }
    with open(path / "data.yaml", "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir-hack-the-planet",
        help="path pointing to the hack the planet dataset",
        default="./data/01_raw/Hack the Planet",
        type=Path,
    )
    parser.add_argument(
        "--input-dir",
        help="path pointing to the yolov8 bbox annotations",
        default="./data/04_feature/beardetection/bearbody/HackThePlanet/",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the annotations",
        default="./data/05_model_input/beardetection/yolov8/",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    return True


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


def random_split(
    annotation_filepaths: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.5,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Returns a dataframe with the following columns:
    - split: str in {train, val, test}
    - filepath: str - filepath pointing to the annotation label
    """
    assert 0.0 <= train_ratio <= 1.0
    assert 0.0 <= val_ratio <= 1.0
    xs = annotation_filepaths.copy()
    n = len(xs)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * (n - train_size))
    random.Random(random_seed).shuffle(xs)
    X_train = xs[:train_size]
    X_val = xs[train_size : train_size + val_size]
    X_test = xs[train_size + val_size :]
    result = []
    for filepath in X_train:
        result.append({"split": "train", "label_filepath": filepath})
    for filepath in X_val:
        result.append({"split": "val", "label_filepath": filepath})
    for filepath in X_test:
        result.append({"split": "test", "label_filepath": filepath})
    return pd.DataFrame(result)


def get_annotation_filepaths(input_dir: Path) -> list[Path]:
    return list(input_dir.rglob("*.txt"))


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        input_dir = args["input_dir"]
        output_dir = args["output_dir"]
        input_dir_hack_the_planet = args["input_dir_hack_the_planet"]
        annotation_filepaths = get_annotation_filepaths(input_dir=input_dir)
        # FIXME: do not random split because of data leak - make it possible to split by date or location
        df_split = random_split(annotation_filepaths=annotation_filepaths)
        df_split["image_filepath"] = df_split["label_filepath"].map(
            lambda label_filepath: label_filepath_to_image_filepath(
                input_dir_hack_the_planet=input_dir_hack_the_planet,
                input_dir=input_dir,
                label_filepath=label_filepath,
            )
        )
        # TODO: remove
        print(df_split.head(n=10))
        output_dir.mkdir(exist_ok=True, parents=True)

        write_data_yaml(output_dir)

        # Copy over the labels and images in the correct folders
        for idx, row in tqdm(df_split.iterrows()):
            split = row["split"]
            src_image_filepath = row["image_filepath"]
            if src_image_filepath:
                dst_image_filepath = (
                    output_dir / split / "images" / src_image_filepath.name
                )
                os.makedirs(dst_image_filepath.parent, exist_ok=True)
                shutil.copy(src=src_image_filepath, dst=dst_image_filepath)
                src_label_filepath = row["label_filepath"]
                dst_label_filepath = (
                    output_dir / split / "labels" / src_label_filepath.name
                )
                os.makedirs(dst_label_filepath.parent, exist_ok=True)
                shutil.copy(src=src_label_filepath, dst=dst_label_filepath)

        exit(0)
