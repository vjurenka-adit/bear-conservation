import logging
import os
import pprint
import shutil
from pathlib import Path

import pandas as pd
import torch

from bearidentification.metriclearning.model.metrics import make_accuracy_calculator
from bearidentification.metriclearning.utils import (
    get_best_device,
    get_best_model_filepath,
    get_transforms,
    load_weights,
    make_dataloaders,
    make_hooks,
    make_model_dict,
    make_tester,
    yaml_write,
)


def accuracies_to_df(accuracies: dict) -> pd.DataFrame:
    """Creates a dataframe that collects all accuracies from the different
    splits."""
    results = []
    for split, metrics in accuracies.items():
        results.append({"split": split, **metrics})
    return pd.DataFrame(results)


def run(
    train_run_root_dir: Path,
    output_dir: Path,
) -> None:
    """Main entrypoint to run the evaluation."""
    device = get_best_device()
    logging.info(f"device: {device}")
    best_model_filepath = get_best_model_filepath(train_run_root_dir=train_run_root_dir)
    assert (
        best_model_filepath.exists()
    ), f"best model filepath does not exist {best_model_filepath}"
    loaded_model = torch.load(
        best_model_filepath,
        map_location=device,
    )
    device = get_best_device()
    args = loaded_model["args"]
    experiment_name = args["run"]["experiment_name"]
    config = args.copy()
    del config["run"]

    transforms = get_transforms(
        data_augmentation=config.get("data_augmentation", {}),
        trunk_preprocessing=config["model"]["trunk"].get("preprocessing", {}),
    )

    logging.info(f"loading the df_split")
    df_split = pd.DataFrame(loaded_model["data_split"])
    df_split.info()

    dataloaders = make_dataloaders(
        batch_size=config["batch_size"],
        df_split=df_split,
        transforms=transforms,
    )

    model_dict = make_model_dict(
        device=device,
        pretrained_backbone=config["model"]["trunk"]["backbone"],
        embedding_size=config["model"]["embedder"]["embedding_size"],
        hidden_layer_sizes=config["model"]["embedder"]["hidden_layer_sizes"],
    )

    trunk_weights = loaded_model["trunk"]
    trunk = model_dict["trunk"]
    trunk = load_weights(
        network=trunk,
        weights=trunk_weights,
        prefix="module.",
    )
    embedder_weights = loaded_model["embedder"]
    embedder = model_dict["embedder"]
    embedder = load_weights(
        network=embedder,
        weights=embedder_weights,
        prefix="module.",
    )

    dataset_train = dataloaders["dataset"]["train"]
    dataset_val = dataloaders["dataset"]["val"]
    dataset_test = dataloaders["dataset"]["test"]

    hooks = make_hooks(
        record_path=output_dir,
        experiment_name=experiment_name,
    )

    accuracy_calculator = make_accuracy_calculator()

    tester = make_tester(
        hooks=hooks,
        record_path=output_dir,
        accuracy_calculator=accuracy_calculator,
    )
    logging.info(f"Initialized the tester {tester}")

    dataset_dict = {}
    dataset_dict["train"] = dataset_train
    if len(dataset_test) > 0:
        dataset_dict["test"] = dataset_test
    dataset_dict["test"] = dataset_test
    if len(dataset_val) > 0:
        dataset_dict["val"] = dataset_val

    splits_to_eval = []
    if "train" in dataset_dict:
        splits_to_eval.append(("train", ["train"]))

    if "val" in dataset_dict:
        splits_to_eval.append(("val", ["val"]))

    if "val" in dataset_dict and "test" in dataset_dict:
        splits_to_eval.append(("test", ["train", "val"]))

    if "val" not in dataset_dict and "test" in dataset_dict:
        splits_to_eval.append(("test", ["test"]))

    logging.info(f"splits_to_eval: {splits_to_eval}")

    all_accuracies = tester.test(
        dataset_dict=dataset_dict,
        epoch=1,
        trunk_model=trunk,
        embedder_model=embedder,
        splits_to_eval=splits_to_eval,
    )

    pprint.pprint(all_accuracies)
    df_metrics = accuracies_to_df(all_accuracies)
    output_filepath = output_dir / "evaluation" / "metrics.csv"
    logging.info(f"Storing metrics in {output_filepath}")
    os.makedirs(output_filepath.parent, exist_ok=True)
    df_metrics.to_csv(output_dir / "evaluation" / "metrics.csv")
    shutil.rmtree(output_dir / "training_logs", ignore_errors=True)
    shutil.rmtree(output_dir / "tensorboard", ignore_errors=True)
    data = {
        **args,
        "eval": {"splits_to_eval": str(splits_to_eval)},
    }
    yaml_write(to=output_dir / "args.yaml", data=data)
