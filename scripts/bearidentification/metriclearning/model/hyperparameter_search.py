import argparse
import logging
import os
import random
from pathlib import Path

from tqdm import tqdm

from bearidentification.metriclearning.model.hyperparameter import (
    grid_space,
    make_config,
)
from bearidentification.metriclearning.model.train import run
from bearidentification.metriclearning.utils import validate_run_config


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        help="Number of configs and runs to make",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--experiment-name",
        help="Name of the experiment",
        type=str,
        default="hyperparameter_search",
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
        "--split-type",
        help="split type, in {by_individual, by_provided_bearid}",
        type=str,
        default="by_provided_bearid",
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
        n = args["n"]
        logging.info(f"Running hyperparameter search for {n} random configs")
        for _ in tqdm(range(n)):
            # Drawing a random seed
            random_seed = random.randint(0, 100000)
            config = make_config(
                random_seed=random_seed,
                grid_space=grid_space,
                split_type=args["split_type"],
            )
            if not validate_run_config(config):
                logging.error(f"The generated config is not valid {config}")
                exit(1)
            else:
                logging.info(f"config file validated {config}")
                try:
                    run(
                        split_root_dir=Path(args["split_root_dir"]),
                        split_type=args["split_type"],
                        dataset_size="full",
                        output_dir=Path(args["save_dir"]),
                        config=config,
                        experiment_name=args["experiment_name"],
                        random_seed=0,
                    )
                except:
                    logging.error(f"Error running the train run with config: {config}")
        exit(0)
