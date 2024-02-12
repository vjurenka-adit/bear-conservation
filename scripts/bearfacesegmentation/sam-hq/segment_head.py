import argparse
import logging
import os
from pathlib import Path

from bearfacesegmentation.sam_hq.segment_head import run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-head-bbox-xml-filepath",
        help="xml filepath containing the annotations and dataset.",
        required=True,
    )
    parser.add_argument(
        "--from-body-masks",
        help="directory containing the bear body masks. Usually generated with SAM HQ.",
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
    if not os.path.isfile(args["from_head_bbox_xml_filepath"]):
        logging.error(
            "invalid --from-head-bbox-xml-filepath -- the filepath does not exist"
        )
        return False
    elif not os.path.isdir(args["bearid_base_path"]):
        logging.error("invalid --bearid_base_path -- the folder does not exist")
        return False
    elif not os.path.isdir(args["from_body_masks"]):
        logging.error("invalid --from-body-masks -- the folder does not exist")
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
        run(
            bearid_base_path=Path(args["bearid_base_path"]),
            label_path=Path(args["from_head_bbox_xml_filepath"]),
            masks_body_dir=Path(args["from_body_masks"]),
            output_dir=Path(args["to"]),
        )
        exit(0)
