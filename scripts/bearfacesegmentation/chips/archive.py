import argparse
import logging
import os
import shutil
from pathlib import Path


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not os.path.isdir(args["source_dir"]):
        logging.error(f'source-dir {args["source_path"]} is not a valid directory.')
        return False
    else:
        return True


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        help="source directory of images to extract the chips from",
        default="./data/01_raw/BearID/images/",
        required=True,
    )
    parser.add_argument(
        "--save-path",
        help="path to save the chips.",
        default="./data/07_model_output/bearfacesegmentation/chips/",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        print(args)
        archive_filepath = Path(args["save_path"]) / "chips_archive"
        logging.info(f"Generating an archive file {archive_filepath}.zip")
        shutil.make_archive(str(archive_filepath), "zip", args["source_dir"])
        logging.info("Done")
        exit(0)
