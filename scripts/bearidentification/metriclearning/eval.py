import argparse
import logging
from pathlib import Path

from bearidentification.metriclearning.eval import run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-run-root-dir",
        help="path to train run",
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


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        run(
            train_run_root_dir=args["train_run_root_dir"],
            output_dir=args["output_dir"],
        )
