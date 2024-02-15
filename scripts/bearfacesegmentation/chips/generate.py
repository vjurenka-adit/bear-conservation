import argparse
import logging
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from bearfacesegmentation.chip import (
    crop_from_yolov8,
    predict_bear_head,
    resize,
    square_pad,
)
from bearfacesegmentation.predict import load_model


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not os.path.isdir(args["source_dir"]):
        logging.error(f'source-dir {args["source_path"]} is not a valid directory.')
        return False
    elif not os.path.isfile(args["instance_segmentation_model_weights"]):
        logging.error(f'model-weights {args["model_weights"]} is not a valid filepath.')
        return False
    else:
        return True


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser for running the download script."""
    parser = argparse.ArgumentParser()
    # TODO: make it possible to use segmented heads from SAM or SAM HQ to generate the chips
    parser.add_argument(
        "--instance-segmentation-model-weights",
        help="model weights filepath of the segmentation model. It should be located in data/06_models.",
        required=True,
    )
    parser.add_argument(
        "--resize-to",
        nargs="+",
        type=int,
        help="width in pixel of the square box for the chip. Multiple arguments can be passed to resize to different dimensions.",
        default=[100, 150, 200, 250, 300],
    )
    parser.add_argument(
        "--source-dir",
        help="source directory of images to extract the chips from",
        default="./data/01_raw/BearID/images/",
        required=True,
    )
    parser.add_argument(
        "--save-path",
        help="path to save the chips.",
        default="./data/07_model_output/bearfacesegmentation/chips/",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def get_filepaths(root: Path, allowed_suffixes={".jpg", ".JPG"}) -> list[Path]:
    """Lists all filepaths given a root directory `root`."""
    return [p for p in root.rglob("*") if p.suffix in allowed_suffixes and p.exists()]


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        missing_bear_heads = []
        source_dir = Path(args["source_dir"])
        save_path = Path(args["save_path"])
        square_dim_list = args["resize_to"]
        image_filepaths = get_filepaths(source_dir)

        logging.info(f"Creating the directory to save the results {save_path}")
        os.makedirs(save_path, exist_ok=True)

        logging.info(
            f"Loading segmentation model from {args['instance_segmentation_model_weights']}"
        )
        model = load_model(
            weights_path=Path(args["instance_segmentation_model_weights"])
        )
        if not model:
            logging.error(
                f"Could not load the model from {args['instance_segmentation_model_weights']}"
            )
            exit(1)

        for image_filepath in tqdm(image_filepaths):
            # To generate the chips, we assume that there is only one bear head
            # per image_filepath
            prediction_yolov8 = predict_bear_head(model, image_filepath, max_det=1)
            if not prediction_yolov8 or not prediction_yolov8.masks:
                missing_bear_heads.append(image_filepath)
                logging.error(f"Can't find bear heads in {image_filepath}")
            else:
                cropped_bear_head = crop_from_yolov8(prediction_yolov8)

                relative = image_filepath.relative_to(source_dir)
                output_raw_chip_filepath = (
                    save_path / "raw" / f"{relative.parent}/{relative.stem}.jpg"
                )
                output_padded_chip_filepath = (
                    save_path / "padded" / f"{relative.parent}/{relative.stem}.jpg"
                )

                padded_cropped_head = square_pad(cropped_bear_head)

                # Creating the directories if needed
                os.makedirs(output_raw_chip_filepath.parent, exist_ok=True)
                os.makedirs(output_padded_chip_filepath.parent, exist_ok=True)

                # Writing the image artifacts
                cv2.imwrite(str(output_raw_chip_filepath), cropped_bear_head)
                cv2.imwrite(str(output_padded_chip_filepath), padded_cropped_head)

                # Creating the thumbnails
                for square_dim in square_dim_list:
                    output_resized_chip_filepath = (
                        save_path
                        / "resized"
                        / f"square_dim_{square_dim}"
                        / f"{relative.parent}/{relative.stem}.jpg"
                    )
                    resized_padded_cropped_head = resize(
                        padded_cropped_head,
                        dim=(square_dim, square_dim),
                    )
                    os.makedirs(output_resized_chip_filepath.parent, exist_ok=True)
                    cv2.imwrite(
                        str(output_resized_chip_filepath), resized_padded_cropped_head
                    )

        if missing_bear_heads:
            logging.warning(
                f"Could not generate chip for the following image_filepaths: {missing_bear_heads}"
            )

        exit(0)
