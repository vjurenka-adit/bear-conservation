import argparse
import logging
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from beardetection.data.augment import augment
from beardetection.data.utils import (
    balance_classes,
    get_ratio_others_over_bears,
    load_datasplit,
    write_data_yaml,
)


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
        "--balance",
        help="value in downsample, upsample - how should we balance the dataset. If nothing is passed, then it does not balance the dataset.",
        required=False,
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


def get_dst_image_filepath(
    output_dir: Path, split: str, src_image_filepath: Path
) -> Path:
    return output_dir / split / "images" / Path(src_image_filepath).name


def copy_over(
    output_dir: Path,
    df_row: pd.Series,
) -> None:
    """Copy over the labels and images in the correct folders."""
    split = df_row["split"]
    src_image_filepath = df_row["image_filepath"]
    m_src_label_filepath = df_row["label_filepath"]

    # dst_image_filepath = output_dir / split / "images" / Path(src_image_filepath).name
    dst_image_filepath = get_dst_image_filepath(
        output_dir=output_dir,
        split=df_row["split"],
        src_image_filepath=Path(src_image_filepath),
    )
    os.makedirs(dst_image_filepath.parent, exist_ok=True)
    shutil.copy(src=src_image_filepath, dst=dst_image_filepath)
    if m_src_label_filepath:
        dst_label_filepath = (
            output_dir / split / "labels" / Path(m_src_label_filepath).name
        )
        os.makedirs(dst_label_filepath.parent, exist_ok=True)
        shutil.copy(src=m_src_label_filepath, dst=dst_label_filepath)


def downsample(df_split: pd.DataFrame, output_dir: Path) -> None:
    """Downsamples the provided df_split to rebalance the classes."""
    df_split_downsampled = balance_classes(df_split)
    logging.info("df_split")
    logging.info(df_split.groupby("class").count())
    logging.info("df_split_downsampled")
    logging.info(df_split_downsampled.groupby("class").count())

    for _, df_row in tqdm(df_split_downsampled.iterrows()):
        copy_over(output_dir=output_dir, df_row=df_row)


def upsample(
    df_split: pd.DataFrame,
    output_dir: Path,
) -> None:
    ratio = get_ratio_others_over_bears(df_split=df_split)
    k = int(round(1.0 / ratio))
    logging.info(f"ratio others / bears: {ratio}")
    logging.info(f"k: {k}")
    for _, df_row in tqdm(df_split.iterrows()):
        copy_over(output_dir=output_dir, df_row=df_row)
        if df_row["class"] == "other" and k > 1:
            image_filepath = Path(df_row["image_filepath"])
            try:
                image = Image.open(image_filepath)
                augmented_images = augment(image)
                images = random.sample(augmented_images, k)
                for i, augmented_image in enumerate(images):
                    dst_image_filepath = get_dst_image_filepath(
                        output_dir=output_dir,
                        split=df_row["split"],
                        src_image_filepath=image_filepath,
                    )
                    augmented_image_filepath = (
                        dst_image_filepath.parent
                        / f"{dst_image_filepath.stem}_{i}{dst_image_filepath.suffix}"
                    )
                    # FIXME
                    logging.info(f"dst_image_filepath: {dst_image_filepath}")
                    logging.info(
                        f"augmented_image_filepath: {augmented_image_filepath}"
                    )
                    augmented_image.save(augmented_image_filepath)
            except:
                logging.error(f"could not augment image: {image_filepath}")


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
        logging.info(f"Loading the datasplit from {args['data_split']}")
        df_split = load_datasplit(args["data_split"])

        logging.info(f"Creating dir at {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        write_data_yaml(output_dir)

        if args["balance"] == "downsample":
            downsample(df_split=df_split, output_dir=output_dir)
        elif args["balance"] == "upsample":
            upsample(df_split=df_split, output_dir=output_dir)
        else:
            for idx, df_row in tqdm(df_split.iterrows()):
                copy_over(output_dir=output_dir, df_row=df_row)

        exit(0)
