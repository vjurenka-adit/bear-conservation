import argparse
import glob
import logging
from pathlib import Path

from beardetection.data.groundingdino import annotate, load_groundingDINO_model
from beardetection.data.utils import get_best_device


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir-hack-the-planet",
        help="path pointing to the hack the planet dataset",
        default="./data/01_raw/Hack the Planet",
        type=Path,
    )
    parser.add_argument(
        "--device",
        help="device to run the model on, can be {auto, cuda, cpu, mps}",
        default="auto",
        type=str,
    )
    parser.add_argument(
        "--model-checkpoint-path",
        help="GroundingDINO model checkpoint path",
        default="./vendors/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the annotations",
        default="./data/07_model_output/beardetection/bearbody/HackThePlanet/",
        type=Path,
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
    if not args["model_checkpoint_path"].exists():
        logging.error(
            f"Invalid --model-checkpoint-path {args['model_checkpoint_path']}"
        )
        return False
    elif args["device"] not in {"auto", "cpu", "cuda", "mps"}:
        logging.error(f"Invalid --device")
        return False
    else:
        return True


SUB_DIRS = [
    "./data/01_raw/Hack the Planet/images/Bison - bears only",
    "./data/01_raw/Hack the Planet/images/Season1 -  bears only",
    "./data/01_raw/Hack the Planet/images/Season2 - bears only",
    "./data/01_raw/Hack the Planet/images/Season3 - bears only",
    "./data/01_raw/Hack the Planet/images/Season5 - bears only",
    "./data/01_raw/Hack the Planet/images/frameOutput/data/BearFeedingPoints - bears only",
    "./data/01_raw/Hack the Planet/images/frameOutput/data/Bison - bears only",
    "./data/01_raw/Hack the Planet/images/frameOutput/data/Season1 -  bears only",
    "./data/01_raw/Hack the Planet/images/frameOutput/data/Season2 - bears only",
    "./data/01_raw/Hack the Planet/images/frameOutput/data/Season3 - bears only",
    "./data/01_raw/Hack the Planet/images/frameOutput/data/Season5 - bears only",
]


def get_image_filepaths(sub_dirs: list[str] = SUB_DIRS) -> list[str]:
    image_extensions = [".jpg", ".PNG", ".JPG"]
    image_paths = [
        path
        for folder in sub_dirs
        for ext in image_extensions
        for path in glob.iglob("%s/**/*%s" % (folder, ext), recursive=True)
    ]
    return image_paths


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)

        image_filepaths = get_image_filepaths(sub_dirs=SUB_DIRS)
        logging.info(f"Retrieved {len(image_filepaths)} image filepaths")
        device = args["device"]
        if device == "auto":
            device = str(get_best_device())

        model = load_groundingDINO_model(
            device=device,
            model_checkpoint_path=args["model_checkpoint_path"],
        )
        text_prompt = "bear"
        token_spans = [[(0, 4)]]
        input_dir = args["input_dir_hack_the_planet"]
        output_dir = args["output_dir"]
        logging.info(
            f"Annotating {len(image_filepaths)} and storing the labels in {output_dir}"
        )
        annotate(
            model=model,
            text_prompt=text_prompt,
            token_spans=token_spans,
            image_paths=image_filepaths,
            input_dir=input_dir,
            output_dir=output_dir,
        )
