import argparse
import glob
import itertools
import logging
import os
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from bearfacedetection.xml_parser import load_xml


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bearid-base-path",
        help="base path of the bearID folder",
        default="./data/01_raw/BearID/",
        required=True,
    )
    parser.add_argument(
        "--to",
        help="directory to store the result of the metadata.",
        default="./data/03_primary/golden_dataset/",
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
    if not os.path.isdir(args["bearid_base_path"]):
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
        base_path = Path(args["bearid_base_path"])
        output_dir = Path(args["to"])

        os.makedirs(output_dir, exist_ok=True)

        xml_filepath_train = base_path / "images_train_without_bc.xml"
        xml_filepath_test = base_path / "images_test_without_bc.xml"
        xml_data_train = load_xml(
            base_path=base_path,
            filepath=xml_filepath_train,
        )
        xml_data_test = load_xml(
            base_path=base_path,
            filepath=xml_filepath_test,
        )

        filepaths_train = [
            image_data["filepath"] for image_data in xml_data_train["images"]
        ]
        filepaths_test = [
            image_data["filepath"] for image_data in xml_data_test["images"]
        ]

        # TODO: investigate why there are fewer unique filenames than filepaths (some misidentified bears in the dataset?)
        all_filenames = {fp.name for fp in [*filepaths_train, *filepaths_test]}

        data = {
            "total_size": len(filepaths_test) + len(filepaths_train),
            "distinct_total_filenames": len(all_filenames),
            "annotations": {
                "train": str(xml_filepath_train),
                "test": str(xml_filepath_test),
            },
            "data_split": {
                "test": {
                    "size": len(filepaths_test),
                    "filepaths": [str(fp) for fp in filepaths_test],
                },
                "train": {
                    "size": len(filepaths_train),
                    "filepaths": [str(fp) for fp in filepaths_train],
                },
            },
            "images": [str(fp) for fp in [*filepaths_train, *filepaths_test]],
        }

        with open(output_dir / "metadata.yaml", "w") as f:
            yaml.dump(
                data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False
            )

        exit(0)
