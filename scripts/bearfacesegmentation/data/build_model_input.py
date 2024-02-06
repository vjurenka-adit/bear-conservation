import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the processor script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-metadata-yaml",
        help="yaml file containing the split information and links to images.",
        default="./data/03_primary/golden_dataset/metadata.yaml",
        required=True,
    )
    parser.add_argument(
        "--yolov8-txt-format",
        help="directory containing the yolov8-txt-format labels",
        default="./data/04_feature/yolov8_txt_format/v0/",
        required=True,
    )

    parser.add_argument(
        "--to",
        help="dir to save the generated model_input",
        required=True,
        default="./data/05_model_input/bearfacesegmentation/golden_dataset/",
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
    if not os.path.isfile(args["split_metadata_yaml"]):
        logging.error(
            "invalid --split-metadata-yaml filepath -- the file does not exist"
        )
        return False
    else:
        return True


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def write_data_yaml(path: Path) -> None:
    """Writes the `data.yaml` file necessary for YOLOv8 training at `path`
    location."""
    data = {
        "train": "./train/images",
        "val": "./val/images",
        "nc": 1,
        "names": ["bearface"],
    }
    with open(path / "data.yaml", "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        print(args)
        output_dir = Path(args["to"])
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "train", exist_ok=True)
        os.makedirs(output_dir / "train" / "images", exist_ok=True)
        os.makedirs(output_dir / "train" / "labels", exist_ok=True)
        os.makedirs(output_dir / "val", exist_ok=True)
        os.makedirs(output_dir / "val" / "images", exist_ok=True)
        os.makedirs(output_dir / "val" / "labels", exist_ok=True)

        with open(args["split_metadata_yaml"], "r") as f:
            print("test2")
            split_metadata = yaml.safe_load(f)
            yolov8_txt_format_path = Path(args["yolov8_txt_format"])
            logging.info(
                f'Building model input for train {split_metadata["data_split"]["train"]["size"]} images'
            )

            logging.info("Writing data.yaml")
            write_data_yaml(output_dir)

            stem_to_train_label_filepath = {
                Path(fp).stem: yolov8_txt_format_path / "train" / fp
                for fp in os.listdir(yolov8_txt_format_path / "train/")
            }
            image_train_dir = output_dir / "train" / "images"
            label_train_dir = output_dir / "train" / "labels"

            for fp in tqdm(split_metadata["data_split"]["train"]["filepaths"]):
                image_filepath = Path(fp)
                label_filepath = stem_to_train_label_filepath.get(image_filepath.stem)
                if label_filepath:
                    shutil.copy(image_filepath, image_train_dir / image_filepath.name)
                    shutil.copy(label_filepath, label_train_dir / label_filepath.name)
                else:
                    logging.warning(
                        f"Missing label_filepath for image: {image_filepath}"
                    )

            logging.info(
                f'Building model input for test {split_metadata["data_split"]["test"]["size"]} images'
            )
            stem_to_test_label_filepath = {
                Path(fp).stem: yolov8_txt_format_path / "test" / fp
                for fp in os.listdir(yolov8_txt_format_path / "test/")
            }
            image_val_dir = output_dir / "val" / "images"
            label_val_dir = output_dir / "val" / "labels"

            for fp in tqdm(split_metadata["data_split"]["test"]["filepaths"]):
                image_filepath = Path(fp)
                label_filepath = stem_to_test_label_filepath.get(image_filepath.stem)
                if label_filepath:
                    shutil.copy(image_filepath, image_val_dir / image_filepath.name)
                    shutil.copy(label_filepath, label_val_dir / label_filepath.name)
                else:
                    logging.warning(
                        f"Missing label_filepath for image: {image_filepath}"
                    )

        exit(0)
