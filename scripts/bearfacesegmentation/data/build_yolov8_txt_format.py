import argparse
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from bearfacesegmentation.data.yolov8_txt_format import (
    mask_filepath_to_yolov8_format_string,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-head-masks",
        help="path containing the head masks",
        required=True,
    )
    parser.add_argument(
        "--bearid-base-path",
        help="BearID base path",
        default="./data/01_raw/BearID/",
    )
    parser.add_argument(
        "--to",
        help="directory to save the processed data. Make sure to use data/04_features.",
        required=True,
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
    if not os.path.isdir(args["from_head_masks"]):
        logging.error("invalid --from-head-masks -- the folder does not exist")
        return False
    elif not os.path.isdir(args["bearid_base_path"]):
        logging.error("invalid --bearid_base_path -- the folder does not exist")
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
        head_masks_dir = Path(args["from_head_masks"])
        masks_filepaths = [
            head_masks_dir / filename for filename in os.listdir(head_masks_dir)
        ]

        output_dir = Path(args["to"])
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Generating the yolov8_txt_format")
        for mask_filepath in tqdm(masks_filepaths):
            content = mask_filepath_to_yolov8_format_string(mask_filepath)
            out_filepath = output_dir / f"{mask_filepath.stem}.txt"
            with open(out_filepath, "w") as f:
                f.write(content)
        exit(0)
