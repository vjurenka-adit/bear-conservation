import argparse
import logging
import os
from pathlib import Path

import bearfacelandmarkdetection.data.build_model_input as build_model_input


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the processor script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        help="directory containing the yolov8 annotations and images.",
        default="./data/04_features/bearfacelandmarkdetection/golden_dataset/",
        required=True,
    )
    parser.add_argument(
        "--to",
        help="dir to save the generated model_input",
        required=True,
        default="./data/05_model_input/bearfacelandmarkdetection/golden_dataset/",
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
    if not os.path.isdir(args["from"]):
        logging.error("invalid --from path -- the directory does not exist")
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
        print(args)
        build_model_input.build_model_input(
            input_dir=Path(args["from"]),
            output_dir=Path(
                args["to"],
            ),
        )
        exit(0)
