import argparse
import logging
import os

from ultralytics import YOLO


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-weights",
        help="The filepath to the yolov8 model weights.",
        default="./data/06_models/beardetection/model/weights/model.pt",
    )
    parser.add_argument(
        "--source-path",
        help="The filepath to the image to run inference on.",
        default="./data/09_external/detect/images/bears/image1.jpg",
    )
    parser.add_argument(
        "--save-path",
        help="The filepath to store the results of the predictions",
        default="./data/07_model_output/beardetection/predictions/",
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
    if not os.path.isfile(args["source_path"]):
        logging.error("invalid --source_path filepath -- the file does not exist")
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
        model_weights = args["model_weights"]
        logging.info(f"loading model from {model_weights}")
        model = YOLO(model_weights)
        logging.info(model.info())
        model.predict(
            args["source_path"],
            save=True,
            project=str(args["save_path"]),
        )
        exit(0)

# ## REPL
# import os
# from pathlib import Path

# model_weights = Path("./data/06_models/beardetection/model/weights/model.pt")
# model = YOLO(model_weights)
# images_dir = Path("./data/05_model_input/beardetection/upsample/yolov8/train/images/")
# images_dir.exists()

# image_filepaths = [images_dir / fp for fp in os.listdir(images_dir)]
# len(image_filepaths)

# batch = image_filepaths[:8]
# batch

# model.predict(batch[0])
# model.predict(batch)
# 4

# 240 / 8
# 180

# 180 / 30

# model.predict(batch[0], imgsz=640)
# model.predict(batch, imgsz=640)

# 73 / 8
# 180 / 9.125
