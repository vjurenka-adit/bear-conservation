import logging
import os
import pprint
import shutil
from pathlib import Path

import pandas as pd

from bearidentification.metriclearning.metrics import make_accuracy_calculator
from bearidentification.metriclearning.utils import (
    get_best_device,
    get_transforms,
    load_datasplit,
    load_weights,
    make_dataloaders,
    make_hooks,
    make_model_dict,
    make_tester,
    yaml_read,
)


# TODO: get rid of predict.py, move some code to eval.py
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
    trunk_weights_filepath = (
        train_run_root_dir / "model" / "weights" / "best" / "trunk.pth"
    )
    embedder_weights_filepath = (
        train_run_root_dir / "model" / "weights" / "best" / "embedder.pth"
    )
    assert (
        trunk_weights_filepath.exists()
    ), f"trunk_filepath does not exist {trunk_weights_filepath}"
    assert (
        embedder_weights_filepath.exists()
    ), f"embedder_filepath does not exist {embedder_weights_filepath}"
    device = get_best_device()
    args_filepath = train_run_root_dir / "args.yaml"
    assert args_filepath.exists(), f"args_filepath does not exist {args_filepath}"

    args = yaml_read(args_filepath)
    experiment_name = args["run"]["experiment_name"]
    dataset_size = args["run"]["datasplit"]["dataset_size"]
    split_type = args["run"]["datasplit"]["split_type"]
    split_root_dir = Path(args["run"]["datasplit"]["split_root_dir"])
    config = args.copy()
    del config["run"]

    transforms = get_transforms(
        transform_type="bare" if "data_augmentation" not in config else "augmented",
        config=config["data_augmentation"] if "data_augmentation" in config else {},
    )

    df_split = load_datasplit(
        split_type=split_type,
        dataset_size=dataset_size,
        split_root_dir=split_root_dir,
    )

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

    trunk = model_dict["trunk"]
    trunk = load_weights(
        network=trunk,
        weights_filepath=trunk_weights_filepath,
        prefix="module.",
    )
    embedder = model_dict["embedder"]
    embedder = load_weights(
        network=embedder,
        weights_filepath=embedder_weights_filepath,
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

    dataset_dict = {"train": dataset_train, "val": dataset_val, "test": dataset_test}
    all_accuracies = tester.test(dataset_dict, 1, trunk, embedder)
    pprint.pprint(all_accuracies)
    df_metrics = accuracies_to_df(all_accuracies)
    output_filepath = output_dir / "evaluation" / "metrics.csv"
    logging.info(f"Storing metrics in {output_filepath}")
    os.makedirs(output_filepath.parent, exist_ok=True)
    df_metrics.to_csv(output_dir / "evaluation" / "metrics.csv")
    shutil.rmtree(output_dir / "training_logs")
    shutil.rmtree(output_dir / "tensorboard")
