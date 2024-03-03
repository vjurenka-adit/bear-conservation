import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
from tqdm.gui import tqdm

import bearfacesegmentation.chip
import bearfacesegmentation.predict
import bearidentification.metriclearning.predict


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        help="k closest embeddings to return.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--metriclearning-args-filepath",
        help="filepath to the yaml file args.yaml used to train the model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metriclearning-embedder-weights-filepath",
        help="weights of the embedder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metriclearning-trunk-weights-filepath",
        help="weights of the trunk",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metriclearning-knn-index-filepath",
        help="filepath to the knn index.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--instance-segmentation-weights-filepath",
        help="model weights filepath. It should be located in data/06_models.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-path",
        help="source path, usually an image filepath or a directory containing images.",
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


# TODO:
def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    return True


def handle_prediction_yolov8(prediction_yolov8, i: int, args: dict) -> None:
    """Handles one predictions_yolov8.

    Running the bearidentification algorithm on the segmented head. It saves in folders specified in `args` the predictions.
    """
    # This is the size used for training the metriclearning
    # model
    SQUARE_DIM = 300
    save_path = args["output_dir"]
    prediction_save_path = save_path / f"prediction_{i}"
    chip_save_path = prediction_save_path / f"chip"
    os.makedirs(chip_save_path, exist_ok=True)
    cropped_bear_head = bearfacesegmentation.chip.crop_from_yolov8(prediction_yolov8)
    padded_cropped_head = bearfacesegmentation.chip.square_pad(cropped_bear_head)
    resized_padded_cropped_head = bearfacesegmentation.chip.resize(
        padded_cropped_head,
        dim=(SQUARE_DIM, SQUARE_DIM),
    )
    cv2.imwrite(str(chip_save_path / "cropped.png"), cropped_bear_head)
    cv2.imwrite(str(chip_save_path / "padded.png"), padded_cropped_head)
    cv2.imwrite(str(chip_save_path / "resized.png"), resized_padded_cropped_head)

    bearidentification.metriclearning.predict.run(
        trunk_weights_filepath=args["metriclearning_trunk_weights_filepath"],
        embedder_weights_filepath=args["metriclearning_embedder_weights_filepath"],
        k=args["k"],
        args_filepath=args["metriclearning_args_filepath"],
        knn_index_filepath=args["metriclearning_knn_index_filepath"],
        chip_filepath=chip_save_path / "resized.png",
        output_dir=prediction_save_path,
    )


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        logging.info(
            f"Loading the model weights from {args['instance_segmentation_weights_filepath']}"
        )
        bearfacesegmentation_model = bearfacesegmentation.predict.load_model(
            weights_path=args["instance_segmentation_weights_filepath"]
        )
        logging.info(f"Running the inference on {args['source_path']}")
        save_path = args["output_dir"]
        image_filepath = args["source_path"]
        if bearfacesegmentation_model:
            predictions_yolov8 = bearfacesegmentation.predict.predict(
                model=bearfacesegmentation_model,
                source=image_filepath,
                save_path=save_path,
            )
            shutil.move(
                src=save_path / "predict" / image_filepath.name,
                dst=save_path / "bearfaces.png",
            )
            shutil.rmtree(save_path / "predict")
            if len(predictions_yolov8) == 0 or not predictions_yolov8[0].masks:
                logging.error(f"Can't find bear heads in {image_filepath}")
            else:
                logging.info(f"{len(predictions_yolov8)} bear faces detected")
                for i, prediction_yolov8 in tqdm(enumerate(predictions_yolov8)):
                    logging.info(f"Identifying face {i}")
                    handle_prediction_yolov8(prediction_yolov8, i=i, args=args)
