import argparse
import logging
import os
from pathlib import Path

from bearidentification.metriclearning.train import run
from bearidentification.metriclearning.utils import validate_run_config, yaml_read


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-seed",
        help="Random Seed to initialize the Random Number Generator",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--experiment-number",
        help="Experiment number",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--experiment-name",
        help="Name of the experiment",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--save-dir",
        help="directory to save the training artefacts.",
        default="./data/06_models/bearidentification/metriclearning/",
    )
    parser.add_argument(
        "--split-root-dir",
        help="root directory containing the data splits.",
        default="data/04_feature/bearidentification/bearid/split/",
    )
    parser.add_argument(
        "--config-file",
        help="filepath to the config for training the training session, it contains all the hyperparameters.",
        default="src/bearidentification/metriclearning/configs/baseline.yaml",
    )
    parser.add_argument(
        "--split-type",
        help="split type, in {by_individual, by_provided_bearid}",
        type=str,
        default="by_provided_bearid",
    )
    parser.add_argument(
        "--dataset-size",
        help="value in {nano, small, medium, large, xlarge, full}",
        type=str,
        default="nano",
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
    if not os.path.isdir(args["split_root_dir"]):
        logging.error(
            "invalid --split-root-dir directory -- the directory does not exist"
        )
        return False
    elif not args["split_type"] in {"by_individual", "by_provided_bearid"}:
        logging.error(
            "invalid --split-type -- the values should be in {by_individual, by_provided_bearid}"
        )
        return False
    elif not args["dataset_size"] in {
        "nano",
        "small",
        "medium",
        "large",
        "xlarge",
        "full",
    }:
        logging.error(
            "invalid --dataset-size -- the values should be in {nano, small, medium, large, xlarge, full}"
        )
        return False
    elif not os.path.isfile(args["config_file"]):
        logging.error("invalid --config-file -- the filepath does not exist.")
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
        print(args)
        config_filepath = Path(args["config_file"])
        logging.info(f"Loading config file {config_filepath}")
        config = yaml_read(config_filepath)
        logging.info(f"config: {config}")
        if not validate_run_config(config):
            logging.error("config file is not valid")
            exit(1)
        else:
            logging.info("config file validated")
            run(
                split_root_dir=Path(args["split_root_dir"]),
                split_type=args["split_type"],
                dataset_size=args["dataset_size"],
                output_dir=Path(args["save_dir"]),
                config=config,
                experiment_name=args["experiment_name"],
                random_seed=args["random_seed"],
            )
        exit(0)
