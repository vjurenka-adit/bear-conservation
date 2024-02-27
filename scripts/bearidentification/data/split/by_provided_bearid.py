import argparse
import logging
import os
from pathlib import Path

from bearidentification.data.split.by_provided_bearid import run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        help="directory to save the generated splits.",
        default="./data/04_feature/bearidentification/bearid/split/",
    )
    parser.add_argument(
        "--chips-root-dir",
        help="root directory containing the BearID chips.",
        default="data/07_model_output/bearfacesegmentation/chips/all/resized/square_dim_300/",
    )
    parser.add_argument(
        "--bearid-root-path",
        help="root directory containing the BearID data.",
        default="data/01_raw/BearID/",
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
    if not os.path.isdir(args["chips_root_dir"]):
        logging.error(
            "invalid --chips-root-dir directory -- the directory does not exist"
        )
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
        run(
            chips_root_dir=Path(args["chips_root_dir"]),
            bearid_root_path=Path(args["bearid_root_path"]),
            save_dir=Path(args["save_path"]),
        )
        exit(0)
