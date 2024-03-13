import argparse
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from beardetection.data.utils import write_data_yaml


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
        "--data-split",
        help="filepath to the csv file containing the datasplit data",
        default="./data/04_feature/beardetection/split/data_split.csv",
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
    if not args["data_split"].exists():
        logging.error(f"Invalid --data-split filepath, it does not exist")
        return False
    else:
        return True


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
        df_split = pd.read_csv(args["data_split"], sep=";")
        logging.info(df_split.head(n=10))
        logging.info(f"Creating dir at {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        write_data_yaml(output_dir)

        # Copy over the labels and images in the correct folders
        for idx, row in tqdm(df_split.iterrows()):
            split = row["split"]
            src_image_filepath = row["image_filepath"]
            if src_image_filepath:
                dst_image_filepath = (
                    output_dir / split / "images" / Path(src_image_filepath).name
                )
                os.makedirs(dst_image_filepath.parent, exist_ok=True)
                shutil.copy(src=src_image_filepath, dst=dst_image_filepath)
                src_label_filepath = row["label_filepath"]
                dst_label_filepath = (
                    output_dir / split / "labels" / Path(src_label_filepath).name
                )
                os.makedirs(dst_label_filepath.parent, exist_ok=True)
                shutil.copy(src=src_label_filepath, dst=dst_label_filepath)

        exit(0)
