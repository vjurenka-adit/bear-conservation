import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

from bearidentification.metriclearning.eval import run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-runs-dir",
        help="path to all train runs",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the eval run metrics.",
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


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    # TODO
    return True


def eval_all(train_runs_dir: Path) -> None:
    """Evaluates all models from train_runs_dir."""
    train_runs_dir = args["train_runs_dir"]
    for train_run_root_dir in tqdm(os.listdir(train_runs_dir)):
        logging.info(f"Evaluating model {train_run_root_dir}")
        run(
            train_run_root_dir=train_runs_dir / train_run_root_dir,
            output_dir=args["output_dir"] / train_run_root_dir,
        )


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        eval_all(train_runs_dir=args["train_runs_dir"])
