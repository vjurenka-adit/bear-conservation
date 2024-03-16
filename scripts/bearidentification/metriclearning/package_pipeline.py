import argparse
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from pytorch_metric_learning.utils.inference import InferenceModel
from torch.utils.data import ConcatDataset

from bearidentification.metriclearning.utils import (
    get_best_device,
    get_transforms,
    load_weights,
    make_dataloaders,
    make_id_mapping,
    make_model_dict,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metriclearning-model-filepath",
        help="filepath to the pytorch metric learning model, usually located in model/best/model.pth",
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
        "--output-filepath",
        help="filepath to save the packaged pipeline.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="dir to save the packaged pipeline.",
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
    if not args["metriclearning_model_filepath"].exists():
        logging.error(
            f"invalid --metriclearning-model-filepath, filepath does not exist"
        )
        return False
    elif not args["instance_segmentation_weights_filepath"].exists():
        logging.error(
            f"invalid --instance-segmentation-weights-filepath, filepath does not exist"
        )
        return False
    else:
        logging.info("Loading metriclearning_model to check keys")
        device = get_best_device()
        logging.info(f"device: {device}")
        loaded_metriclearning_model = torch.load(
            args["metriclearning_model_filepath"],
            map_location=device,
        )
        expected_keys = {"trunk", "embedder", "data_split", "args"}
        keys = {k for k in loaded_metriclearning_model.keys()}
        if not keys == expected_keys:
            logging.error(f"{keys} are different from {expected_keys}")
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
        instance_segmentation_weights = torch.load(
            args["instance_segmentation_weights_filepath"]
        )
        device = get_best_device()
        bearidentification_model = torch.load(
            args["metriclearning_model_filepath"],
            map_location=device,
        )
        df_split = pd.DataFrame(bearidentification_model["data_split"])
        chips_root_dir = Path("/".join(df_split.iloc[0]["path"].split("/")[:-4]))
        logging.info(f"chips_root_dir found: {chips_root_dir}")
        assert chips_root_dir.exists(), "Should find the chips root dir"

        # Creating a temporary folder to prepare the archive
        zip_dir = Path("/tmp/packaged_pipeline_zip/")
        shutil.rmtree(zip_dir, ignore_errors=True)
        os.makedirs(zip_dir, exist_ok=True)
        os.makedirs(zip_dir / "bearfacesegmentation", exist_ok=True)
        os.makedirs(zip_dir / "bearidentification", exist_ok=True)
        os.makedirs(zip_dir / "chips", exist_ok=True)

        logging.info(f"Copying chips from {chips_root_dir}")
        shutil.copytree(
            src=chips_root_dir,
            dst=zip_dir / "chips",
            dirs_exist_ok=True,
        )
        shutil.copy(
            src=args["instance_segmentation_weights_filepath"],
            dst=zip_dir / "bearfacesegmentation" / "model.pt",
        )
        shutil.copy(
            src=args["metriclearning_model_filepath"],
            dst=zip_dir / "bearidentification" / "model.pt",
        )

        config = bearidentification_model["args"]
        id_mapping = make_id_mapping(df=df_split)
        transforms = get_transforms(
            data_augmentation=config.get("data_augmentation", {}),
            trunk_preprocessing=config["model"]["trunk"].get("preprocessing", {}),
        )
        dataloaders = make_dataloaders(
            batch_size=config["batch_size"],
            df_split=df_split,
            transforms=transforms,
        )
        device = get_best_device()
        model_dict = make_model_dict(
            device=device,
            pretrained_backbone=config["model"]["trunk"]["backbone"],
            embedding_size=config["model"]["embedder"]["embedding_size"],
            hidden_layer_sizes=config["model"]["embedder"]["hidden_layer_sizes"],
        )
        trunk_weights = bearidentification_model["trunk"]
        trunk = model_dict["trunk"]
        trunk = load_weights(
            network=trunk,
            weights=trunk_weights,
            prefix="module.",
        )
        embedder_weights = bearidentification_model["embedder"]
        embedder = model_dict["embedder"]
        embedder = load_weights(
            network=embedder,
            weights=embedder_weights,
            prefix="module.",
        )
        metriclearning_inference_model = InferenceModel(
            trunk=trunk,
            embedder=embedder,
        )
        dataset_full = dataloaders["dataset"]["full"]

        logging.info("Training KNN on the full dataset...")
        metriclearning_inference_model.train_knn(dataset_full)
        knn_index_filepath = zip_dir / "bearidentification" / "knn.index"
        metriclearning_inference_model.knn_func.save(filename=str(knn_index_filepath))
        base_name, format = "__packaged_pipeline", "zip"
        shutil.make_archive(
            base_name=base_name,
            format=format,
            root_dir=zip_dir,
        )
        output_dir = args["output_dir"]
        src_filepath = f"{base_name}.{format}"
        dst_filepath = output_dir / f"packaged_pipeline.zip"
        logging.info(f"Saving the packaged pipeline in {dst_filepath}")
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(src=src_filepath, dst=dst_filepath)
        os.remove(src_filepath)

## REPL Driven

# metriclearning_model_filepath = Path(
#     "./data/06_models/bearidentification/metric_learning/baseline_circleloss_nano_by_provided_bearid/model/weights/best/model.pth"
# )
# bearidentification_model = torch.load(metriclearning_model_filepath)
# bearidentification_model.keys()

# df_split = pd.DataFrame(bearidentification_model["data_split"])
# chips_root_dir = Path("/".join(df_split.iloc[0]["path"].split("/")[:-4]))
# chips_root_dir.exists()

# tmp_dir = Path("/tmp/chips_test_0")
# os.makedirs(tmp_dir, exist_ok=True)

# shutil.copytree(
#     src=chips_root_dir,
#     dst=tmp_dir,
#     dirs_exist_ok=True,
# )
