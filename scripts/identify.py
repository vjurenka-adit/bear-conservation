import argparse
import logging
import os
from pathlib import Path

import cv2
from joblib.logger import shutil

import bearfacedetection
import bearfacesegmentation.predict
import bearidentification
import bearidentification.metriclearning.predict
from bearfacesegmentation.chip import (
    crop_from_yolov8,
    predict_bear_head,
    resize,
    square_pad,
)


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


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        print(args)
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
            # TODO: only call the model once instead of twice to save
            # the segmented face
            bearfacesegmentation.predict.predict(
                model=bearfacesegmentation_model,
                source=image_filepath,
                save_path=save_path,
            )
            shutil.move(
                src=save_path / "predict" / image_filepath.name,
                dst=save_path / "bearfacesegmentation.png",
            )
            shutil.rmtree(save_path / "predict")
            prediction_yolov8 = predict_bear_head(
                model=bearfacesegmentation_model,
                image_filepath=image_filepath,
                max_det=1,
            )
            if not prediction_yolov8 or not prediction_yolov8.masks:
                logging.error(f"Can't find bear heads in {image_filepath}")
            else:
                chip_save_path = save_path / "chip"
                os.makedirs(chip_save_path, exist_ok=True)
                cropped_bear_head = crop_from_yolov8(prediction_yolov8)
                padded_cropped_head = square_pad(cropped_bear_head)
                # This is the size used for training the metriclearning
                # model
                square_dim = 300
                resized_padded_cropped_head = resize(
                    padded_cropped_head,
                    dim=(square_dim, square_dim),
                )
                cv2.imwrite(str(chip_save_path / "cropped.png"), cropped_bear_head)
                cv2.imwrite(str(chip_save_path / "padded.png"), padded_cropped_head)
                cv2.imwrite(
                    str(chip_save_path / "resized.png"), resized_padded_cropped_head
                )
                bearidentification.metriclearning.predict.run(
                    trunk_weights_filepath=args[
                        "metriclearning_trunk_weights_filepath"
                    ],
                    embedder_weights_filepath=args[
                        "metriclearning_embedder_weights_filepath"
                    ],
                    k=args["k"],
                    args_filepath=args["metriclearning_args_filepath"],
                    knn_index_filepath=args["metriclearning_knn_index_filepath"],
                    chip_filepath=chip_save_path / "resized.png",
                    output_dir=args["output_dir"],
                )
