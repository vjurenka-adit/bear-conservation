import logging
import os
import shutil
from pathlib import Path

import yaml


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


def build_model_input(input_dir: Path, output_dir: Path):
    dirs = [
        output_dir / "train/images/",
        output_dir / "train/labels/",
        output_dir / "val/images/",
        output_dir / "val/labels/",
    ]

    for dir in dirs:
        if not os.path.isdir(dir):
            logging.info(f"Making directory: {dir}")
            os.makedirs(dir)

    logging.info("Writing data.yaml file")
    write_data_yaml(output_dir)

    logging.info("Copying files over")
    shutil.copytree(input_dir / "train", output_dir / "train", dirs_exist_ok=True)
    shutil.copytree(input_dir / "test", output_dir / "val", dirs_exist_ok=True)
