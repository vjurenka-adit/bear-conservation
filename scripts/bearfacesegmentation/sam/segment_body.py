import argparse
import logging
import os
from pathlib import Path

from bearfacesegmentation.sam.segment_body import load_sam_predictor, run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml-filepath",
        help="xml filepath containing the annotations and dataset.",
        required=True,
    )
    parser.add_argument(
        "--model-weights",
        help="filepath containing the sam weights.",
        default="./data/06_models/bearfacesegmentation/sam/weights/sam_vit_h_4b8939.pth",
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
    if not os.path.isfile(args["xml_filepath"]):
        logging.error("invalid --xml-filepath -- the file does not exist")
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
        print(args)
        logging.info("Loading the SAM model")
        sam_predictor = load_sam_predictor(
            sam_checkpoint_path=Path(args["model_weights"])
        )
        run(
            predictor=sam_predictor,
            base_path=Path(args["bearid_base_path"]),
            label_path=Path(args["xml_filepath"]),
            output_dir=Path(args["to"]),
        )
        exit(0)
