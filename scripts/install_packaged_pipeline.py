import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

from bearidentification.metriclearning.utils import get_best_device


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packaged-pipeline-archive-filepath",
        help="filepath to the packaged pipeline, usually as a zip file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def md5(filepath: Path) -> str:
    """Returns the md5 hash of a filepath."""
    with open(filepath, "rb") as f:
        file_contents = f.read()
        return hashlib.md5(file_contents).hexdigest()


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["packaged_pipeline_archive_filepath"].exists():
        logging.error(
            f"invalid --packaged-pipeline-archive-filepath, filepath does not exist"
        )
        return False
    else:
        MD5_VERIFIED_HASHES = {
            "5758de3c2815db4b92f1513d95f0b62b",
            "16547552d05d189090cb5a269b6555ee",
        }
        md5_hash = md5(args["packaged_pipeline_archive_filepath"])
        logging.info(f"archive md5 hash: {md5_hash}")
        if md5_hash not in MD5_VERIFIED_HASHES:
            logging.error("invalid packaged-pipeline, md5 hashes do not match")
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
        # TODO: add a md5 check here
        logging.info(args)
        device = get_best_device()

        INSTALL_PATH = Path("./data/06_models/pipeline/metriclearning/")
        logging.info(f"Installing the packaged pipeline in {INSTALL_PATH}")
        os.makedirs(INSTALL_PATH, exist_ok=True)
        packaged_pipeline_archive_filepath = args["packaged_pipeline_archive_filepath"]
        shutil.unpack_archive(
            filename=packaged_pipeline_archive_filepath,
            extract_dir=INSTALL_PATH,
        )
        metriclearning_model_filepath = INSTALL_PATH / "bearidentification" / "model.pt"
        bearidentification_model = torch.load(
            metriclearning_model_filepath,
            map_location=device,
        )
        df_split = pd.DataFrame(bearidentification_model["data_split"])

        chips_root_dir = Path("/".join(df_split.iloc[0]["path"].split("/")[:-4]))
        logging.info(f"Retrieved chips_root_dir: {chips_root_dir}")
        os.makedirs(chips_root_dir, exist_ok=True)
        shutil.copytree(
            src=INSTALL_PATH / "chips",
            dst=chips_root_dir,
            dirs_exist_ok=True,
        )
        exit(0)
