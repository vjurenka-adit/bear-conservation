import argparse
import logging
from pathlib import Path

from bearidentification.metriclearning.predict import run


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        help="k closest embeddings to return.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--args-filepath",
        help="filepath to the yaml file args.yaml used to train the model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedder-weights-filepath",
        help="weights of the embedder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trunk-weights-filepath",
        help="weights of the trunk",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--knn-index-filepath",
        help="filepath to the knn index.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--chip-filepath",
        help="filepath to the image chip to run predictions on.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the predictions.",
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
        print(args)
        run(
            trunk_weights_filepath=args["trunk_weights_filepath"],
            embedder_weights_filepath=args["embedder_weights_filepath"],
            k=args["k"],
            args_filepath=args["args_filepath"],
            knn_index_filepath=args["knn_index_filepath"],
            chip_filepath=args["chip_filepath"],
            output_dir=args["output_dir"],
        )
