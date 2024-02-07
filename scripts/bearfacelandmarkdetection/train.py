"""Finetuning YOLOv8.

This script allows the user to fine tune some YOLOv8 models. It requires
the data in the YOLOv8 Pytorch TXT format to be provided.
"""
import argparse
import logging
import os
from pathlib import Path

from bearfacelandmarkdetection.train import load_model, train


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        help="Name of the training experiment. A folder with this name is created to store the model artifacts.",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="pretrained model to use for finetuning. eg. yolov8n.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size: number of images per batch.",
        default=16,
    )
    parser.add_argument(
        "--degrees",
        type=int,
        help="data augmentation: random degree rotation in 0-degree.",
        default=0,
    )
    parser.add_argument(
        "--translate",
        type=float,
        help="data augmentation: random translation.",
        default=0.1,
    )
    parser.add_argument(
        "--flipud",
        type=float,
        help="data augmentation: flip upside down probability.",
        default=0.0,
    )
    parser.add_argument(
        "--data",
        help="The data.yaml file containing information for yolov8 to train.",
        default="data/05_model_input/bearfacelandmarkdetection/golden_dataset/data.yaml",
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
    if not os.path.isfile(args["data"]):
        logging.error("invalid --data filepath -- the file does not exist")
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
        logging.info(f"training started with the following args: {args}")
        model = load_model(args["model"])
        train(model, args)
        exit(0)
