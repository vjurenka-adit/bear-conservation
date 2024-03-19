import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

from beardetection.model.eval import load_model, run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root-dir",
        help="The root dir containing all the trained models.",
        default="data/06_models/beardetection/yolov8/",
        type=Path,
    )
    parser.add_argument(
        "--data",
        help="The data.yaml file containing information for yolov8 to train.",
        default="data/05_model_input/beardetection/upsample_test_cleaned/yolov8/data.yaml",
        # default="data/05_model_input/beardetection/upsample/yolov8/data.yaml",
        type=Path,
    )
    parser.add_argument(
        "--save-path",
        help="The dir to store the results of the predictions",
        default="./data/08_reporting/beardetection/yolov8/evaluation/",
        type=Path,
    )
    parser.add_argument(
        "--split",
        help="The split to evaluate on. In {train, val, test}",
        default="test",
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
    if args["split"] not in ["train", "val", "test"]:
        logging.error("invalid --split value -- Should be in {train, val, test}")
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
        models_root_dir = args["models_root_dir"]
        experiment_names = [
            dir
            for dir in os.listdir(models_root_dir)
            if (models_root_dir / dir).is_dir()
        ]
        logging.info(
            f"Found {len(experiment_names)} trained models: {experiment_names}"
        )
        for experiment_name in tqdm(experiment_names):
            try:
                logging.info(f"Evaluating model: {experiment_name}")
                model_weights = (
                    models_root_dir / experiment_name / "weights" / "best.pt"
                )
                logging.info(f"loading model from {model_weights}")
                model = load_model(model_weights)
                logging.info(model.info())
                data_filepath = args["data"]
                save_path = args["save_path"] / experiment_name
                split = args["split"]
                logging.info(f"Running evaluation on split {split}")
                run(model, data_filepath, split=split, save_path=save_path)
            except Exception as e:
                logging.error(f"error: {e}")
        exit(0)
