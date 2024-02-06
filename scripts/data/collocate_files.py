import argparse
import glob
import itertools
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bearid-images-path",
        help="path of the bearID folder",
        default="./data/01_raw/BearID/images/",
        required=True,
    )
    parser.add_argument(
        "--to",
        help="directory to store the flat images",
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
    if not os.path.isdir(args["bearid_images_path"]):
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
        extensions = ["jpg", "JPG", "PNG", "png"]
        input_dir = Path(args["bearid_images_path"])
        output_dir = Path(args["to"])
        all_nested_files = [
            glob.glob(str(input_dir) + f"/**/*.{extension}", recursive=True)
            for extension in extensions
        ]
        all_filepaths = [Path(fp) for fp in list(itertools.chain(*all_nested_files))]
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Copying all filepaths to {output_dir}")
        for filepath in tqdm(all_filepaths):
            shutil.copy(filepath, output_dir / filepath.name)

        exit(0)
