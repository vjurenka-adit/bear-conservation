import argparse
import logging
from pathlib import Path

from tqdm import tqdm


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help="path pointing to the bbox annotations from GroundingDINO",
        default="./data/07_model_output/beardetection/bearbody/HackThePlanet/",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the annotations",
        default="./data/04_feature/beardetection/bearbody/HackThePlanet/",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def get_annotation_filepaths(input_dir: Path) -> list[Path]:
    return list(input_dir.rglob("*.txt"))


def write_yolov8_annotation(
    annotation_filepath: Path,
    input_dir: Path,
    output_dir: Path,
    bearbody_class: int = 0,
) -> None:
    with open(annotation_filepath, "r") as f:
        content_bbox = f.read()
        content_with_class = f"{bearbody_class} {content_bbox}"
        output_filepath = output_dir / annotation_filepath.relative_to(input_dir)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w") as g:
            g.write(content_with_class)


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        annotation_filepaths = get_annotation_filepaths(input_dir=args["input_dir"])
        logging.info(f"found {len(annotation_filepaths)} annotation filepaths")
        input_dir = args["input_dir"]
        output_dir = args["output_dir"]
        for annotation_filepath in tqdm(annotation_filepaths):
            write_yolov8_annotation(
                annotation_filepath=annotation_filepath,
                input_dir=input_dir,
                output_dir=output_dir,
            )
