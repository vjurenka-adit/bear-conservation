import argparse
import logging
from pathlib import Path

from bearfacesegmentation.predict import load_model, predict


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not Path(args["source_path"]).exists():
        logging.error(f'source-path {args["source_path"]} does not exist')
        return False
    else:
        return True


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-weights",
        help="model weights filepath. It should be located in data/06_models.",
        required=True,
    )
    parser.add_argument(
        "--source-path",
        help="source path, usually an image filepath or a directory containing images.",
        default="data/05_model_input/golden_dataset/val/images/",
        required=True,
    )
    parser.add_argument(
        "--save-path",
        help="path to save the predictions made by the model. Usually data/08_model_output/bearfacedetection/golden_dataset/.",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(f"Loading the model weights from {args['model_weights']}")
        model = load_model(weights_path=args["model_weights"])
        logging.info(f"Running the inference on {args['source_path']}")
        if model:
            if args["save_path"]:
                predict(
                    model=model, source=args["source_path"], save_path=args["save_path"]
                )
            else:
                predict(model=model, source=args["source_path"])
            logging.info("Done")
        exit(0)
