import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, samplers, trainers

from bearidentification.data.split.utils import THRESHOLDS
from bearidentification.metriclearning.model.metrics import make_accuracy_calculator
from bearidentification.metriclearning.utils import (
    BearDataset,
    fix_random_seed,
    get_best_device,
    get_best_model_filepath,
    get_best_weights,
    get_best_weights_filepaths,
    get_transforms,
    load_datasplit,
    make_dataloaders,
    make_hooks,
    make_id_mapping,
    make_model_dict,
    make_tester,
    resize_dataframe,
    save_sample_batch,
    yaml_read,
    yaml_write,
)


def next_experiment_number(experiment_name: str, output_dir: Path) -> int:
    """Returns a natural number corresponding to the next experiment number."""
    if not (output_dir / experiment_name).exists():
        return 0
    else:
        similar_experiment_paths = output_dir.glob(f"{experiment_name}_*")
        similar_experiment_names = [
            d.name for d in similar_experiment_paths if d.is_dir()
        ]
        suffixes = [int(name.split("_")[-1]) for name in similar_experiment_names]
        experiment_number = max([0, *suffixes]) + 1
        return experiment_number


def get_record_path(experiment_name: str, output_dir: Path) -> Path:
    """Returns the path where all the artefacts will be logged."""
    experiment_number = next_experiment_number(
        experiment_name=experiment_name,
        output_dir=output_dir,
    )
    if experiment_number == 0:
        return output_dir / experiment_name
    else:
        return output_dir / f"{experiment_name}_{experiment_number}"


def make_embedder_optimizer(config: dict, embedder: nn.Module) -> torch.optim.Optimizer:
    """Makes the embedder optimizer."""
    embedder_config = config["config"]
    if config["type"] == "adam":
        return torch.optim.Adam(
            embedder.parameters(),
            lr=embedder_config["lr"],
            weight_decay=embedder_config.get("weight_decay", 0.0),
        )
    else:
        raise Exception("The only supported optimizer type is Adam for now.")


def make_trunk_optimizer(config: dict, trunk: nn.Module) -> torch.optim.Optimizer:
    """Makes the trunk optimizer."""
    if config["type"] == "adam":
        trunk_config = config["config"]
        return torch.optim.Adam(
            trunk.parameters(),
            lr=trunk_config["lr"],
            weight_decay=trunk_config.get("weight_decay", 0.0),
        )
    else:
        raise Exception("The only supported optimizer type is Adam for now.")


def make_losses_optimizers(
    config: dict, loss_funcs: dict
) -> dict[str, torch.optim.Optimizer]:
    """Makes the losses optimizers.

    Sometimes, losses have learnable parameters. One example is
    ArcFaceLoss.
    """
    result = {}
    for loss_type, optimizer_config in config.items():
        assert (
            loss_type in loss_funcs
        ), f"loss type {loss_type} is not a key in {loss_funcs}"
        if optimizer_config["type"] == "adam":
            optimizer = torch.optim.Adam(
                loss_funcs[loss_type].parameters(),
                lr=optimizer_config["config"]["lr"],
                weight_decay=optimizer_config["config"]["weight_decay"],
            )
            result[f"{loss_type}_optimizer"] = optimizer
        else:
            raise Exception("The only supported optimizer type is Adam for now.")
    return result


def make_optimizers(
    config: dict,
    trunk: nn.Module,
    embedder: nn.Module,
    loss_funcs: dict,
) -> dict:
    """Returns a dictionnary and one key for the embedder and one for trunk.

    The currently supported optimizer types are {adam}.
    """

    embedder_optimizer = make_embedder_optimizer(
        config=config["embedder"],
        embedder=embedder,
    )
    trunk_optimizer = make_trunk_optimizer(
        config=config["trunk"],
        trunk=trunk,
    )
    losses_optimizers = make_losses_optimizers(
        config=config.get("losses", {}),
        loss_funcs=loss_funcs,
    )

    return {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
        **losses_optimizers,
    }


def make_loss_functions(loss_type: str, config: dict) -> dict:
    """Returns a dictionnary with the metric_loss key.

    Currently supported loss_type in {circleloss}.
    """
    if loss_type == "circleloss":
        metric_loss = losses.CircleLoss(
            m=config["m"],
            gamma=config["gamma"],
        )
        return {"metric_loss": metric_loss}
    elif loss_type == "tripletmarginloss":
        metric_loss = losses.TripletMarginLoss(margin=config["margin"])
        return {"metric_loss": metric_loss}
    elif loss_type == "arcfaceloss":
        metric_loss = losses.ArcFaceLoss(
            num_classes=config["num_classes"],
            embedding_size=config["embedding_size"],
            margin=config["margin"],
            scale=config["scale"],
        )
        return {"metric_loss": metric_loss}
    else:
        raise Exception(f"loss_type {loss_type} not yet implemented")


def make_mining_functions(miner_type: str, config: dict) -> dict:
    """Returns a dictionnary with the key tuple_miner.

    Currently supported miner types in {batcheasyhardminer}.
    """
    if miner_type == "batcheasyhardminer":
        tuple_miner = miners.BatchEasyHardMiner(
            pos_strategy=config["pos_strategy"],
            neg_strategy=config["neg_strategy"],
        )
        return {"tuple_miner": tuple_miner}
    else:
        raise Exception(
            f"Can't initialize mining functions for miner_type {miner_type}"
        )


def make_sampler(
    sampler_type: Optional[str],
    id_mapping: pd.DataFrame,
    train_dataset: BearDataset,
    config: dict,
):
    """Returns a Sampler using the config. Returns None if the sampler_type is
    None.

    Supported sampler types in {mperclass}.
    """
    if not sampler_type:
        return None
    elif sampler_type == "mperclass":
        label_to_bear_id = id_mapping["label"].to_dict()
        bear_id_to_label = {v: k for k, v in label_to_bear_id.items()}
        labels = (
            train_dataset.dataframe["bear_id"]
            .map(lambda bear_id: bear_id_to_label[bear_id])
            .to_list()
        )
        return samplers.MPerClassSampler(
            labels=labels,
            m=config["m"],
            length_before_new_iter=len(train_dataset),
        )
    else:
        raise Exception(f"sampler_type {sampler_type} not implemented.")


def make_end_of_epoch_hook(
    hooks,
    tester,
    dataset_dict: dict,
    model_folder: Path,
    patience: int = 10,
    test_interval: int = 1,
):
    """Returns an end of epoch hook that runs after every end of epoch.

    args:
    - dataset_dict: a dictionnary {'train': train_dataset, 'val':
    val_dataset} mapping the split to the dataset. It is used for evaluation by the provided `tester`.
    - patience: How long do we wait until we stop training (when the validation metrics are not improving after number of epochs.
    - test_interval: how often do we test? After each epoch?
    """
    return hooks.end_of_epoch_hook(
        tester,
        dataset_dict,
        model_folder,
        test_interval=test_interval,
        patience=patience,
    )


def make_trainer(
    model_dict: dict,
    optimizers: dict,
    batch_size: int,
    loss_funcs: dict,
    train_dataset: BearDataset,
    mining_funcs: Optional[dict],
    sampler,
    end_of_iteration_hook,
    end_of_epoch_hook,
):
    """Makes a trainer."""
    return trainers.MetricLossOnly(
        models=model_dict,
        optimizers=optimizers,
        batch_size=batch_size,
        loss_funcs=loss_funcs,
        dataset=train_dataset,
        mining_funcs=mining_funcs,
        sampler=sampler,
        dataloader_num_workers=2,
        end_of_iteration_hook=end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )


def train(trainer, num_epochs: int):
    """Run the provided `trainer` for `num_epochs`."""
    return trainer.train(num_epochs=num_epochs)


def save_best_model_weights(model_folder: Path, output_dir: Path) -> None:
    """Saves the best model weights in the specified `output_dir`."""
    best_weights_filepaths = get_best_weights_filepaths(model_folder=model_folder)
    logging.info(f"Copying best model weights in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(src=best_weights_filepaths["embedder"], dst=output_dir / "embedder.pth")
    shutil.copy(src=best_weights_filepaths["trunk"], dst=output_dir / "trunk.pth")


def package_model(train_run_root_dir: Path) -> dict:
    """Packages in the same python dict the following artifacts:

    - best trunk_weights: dict
    - best embedder_weights: dict
    - args: dict that fully describes the run configuration and hyperparameters
    - data_split: dict that can be loaded as a pandas dataframe. The data split used for the run.

    It makes it easy to ship the model around and to only point to one filepath when loading it.

    args:
    - train_run_root_dir: Path - directory of the train_run
    """
    model_folder = train_run_root_dir / "model"
    assert model_folder.exists(), f"model folder does not exist: {model_folder}"

    args_filepath = train_run_root_dir / "args.yaml"
    assert args_filepath.exists(), f"args_filepath does not exist {args_filepath}"

    args = yaml_read(args_filepath)
    best_weights = get_best_weights(model_folder=model_folder)

    datasplit = args["run"]["datasplit"]

    df_split = load_datasplit(
        split_type=datasplit["split_type"],
        dataset_size=datasplit["dataset_size"],
        split_root_dir=Path(datasplit["split_root_dir"]),
    )
    return {
        "args": args,
        **best_weights,
        "data_split": df_split.to_dict(),
    }


def run(
    split_root_dir: Path,
    split_type: str,
    dataset_size: str,
    output_dir: Path,
    config: dict,
    experiment_name: str,
    random_seed: int = 0,
) -> None:
    """Main entrypoint to setup a training pipeline based on the provided
    config.

    See src/bearidentification/metriclearning/configs/ for configs
    examples.
    """
    logging.info(f"Fixing random_seed to {random_seed}")
    fix_random_seed(random_seed)

    logging.info(f"Setting up the experiment {experiment_name}")
    record_path = get_record_path(
        experiment_name=experiment_name,
        output_dir=output_dir,
    )
    logging.info(f"record_path: {record_path}")
    device = get_best_device()
    logging.info(f"Acquired the device {device}")
    logging.info(
        f"Loading datasplit from split_type {split_type} and size {dataset_size}"
    )
    df_split = load_datasplit(
        split_type=split_type,
        dataset_size=dataset_size,
        split_root_dir=split_root_dir,
    )

    transforms = get_transforms(
        data_augmentation=config.get("data_augmentation", {}),
        trunk_preprocessing=config["model"]["trunk"].get("preprocessing", {}),
    )
    logging.info(f"Loaded the transforms: {transforms}")

    dataloaders = make_dataloaders(
        batch_size=config["batch_size"],
        df_split=df_split,
        transforms=transforms,
    )

    threshold_value = THRESHOLDS["small"]
    df_split_small = resize_dataframe(df=df_split, threshold_value=threshold_value)
    dataloaders_small = make_dataloaders(
        batch_size=config["batch_size"],
        df_split=df_split_small,
        transforms=transforms,
    )
    logging.info(f"df_split_small info:")
    print(df_split_small.info(verbose=True))

    save_sample_batch(
        dataloader=dataloaders["loader"]["viz"], to=record_path / "batch_0.png"
    )

    hooks = make_hooks(record_path=record_path, experiment_name=experiment_name)

    accuracy_calculator = make_accuracy_calculator()
    tester = make_tester(
        hooks=hooks,
        record_path=record_path,
        accuracy_calculator=accuracy_calculator,
    )
    logging.info(f"Initialized the tester {tester}")

    model_dict = make_model_dict(
        device=device,
        pretrained_backbone=config["model"]["trunk"]["backbone"],
        embedding_size=config["model"]["embedder"]["embedding_size"],
        hidden_layer_sizes=config["model"]["embedder"]["hidden_layer_sizes"],
    )
    logging.info(f"Initialized the model_dict {model_dict}")

    loss_funcs = make_loss_functions(
        loss_type=config["loss"]["type"],
        config=config["loss"]["config"],
    )
    logging.info(f"Initialized the loss_funcs: {loss_funcs}")

    optimizers = make_optimizers(
        config=config["optimizers"],
        trunk=model_dict["trunk"],
        embedder=model_dict["embedder"],
        loss_funcs=loss_funcs,
    )
    logging.info(f"Initialized the optimizers {optimizers}")

    mining_funcs = None
    if "miner" in config:
        mining_funcs = make_mining_functions(
            miner_type=config["miner"]["type"],
            config=config["miner"]["config"],
        )
        logging.info(f"Initialized the mining_funcs: {mining_funcs}")

    id_mapping = make_id_mapping(df=df_split)

    sampler = None

    if "sampler" in config:
        sampler = make_sampler(
            sampler_type=config["sampler"]["type"],
            id_mapping=id_mapping,
            train_dataset=dataloaders["dataset"]["train"],
            config=config["sampler"]["config"],
        )
        logging.info(f"Initialized the sampler: {sampler}")

    # We use the test set instead of the val set when the val set is not available
    # This is the case in the by_individual split
    len_val = len(dataloaders["dataset"]["val"])
    logging.info(f"{len_val} datapoints in validation set")
    if len_val == 0:
        logging.info("Using the test set instead of the validation set")
    dataset_val = (
        dataloaders["dataset"]["test"]
        if len_val == 0
        else dataloaders["dataset"]["val"]
    )

    dataset_val_small = (
        dataloaders_small["dataset"]["test"]
        if len_val == 0
        else dataloaders_small["dataset"]["val"]
    )

    dataset_dict = {
        "train": dataloaders["dataset"]["train"],
        "train_small": dataloaders_small["dataset"]["train"],
        "val": dataset_val,
        "val_small": dataset_val_small,
    }

    model_folder = record_path / "model"
    end_of_epoch_hook = make_end_of_epoch_hook(
        hooks=hooks,
        tester=tester,
        dataset_dict=dataset_dict,
        model_folder=model_folder,
        patience=config["patience"],
        test_interval=1,
    )

    trainer = make_trainer(
        model_dict=model_dict,
        optimizers=optimizers,
        batch_size=config["batch_size"],
        loss_funcs=loss_funcs,
        train_dataset=dataloaders["dataset"]["train"],
        mining_funcs=mining_funcs,
        sampler=sampler,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    logging.info(f"Initialized the trainer {trainer}")
    run_config_filepath = record_path / "args.yaml"
    args = config.copy()
    args_run = {
        "random_seed": random_seed,
        "experiment_name": experiment_name,
        "datasplit": {
            "dataset_size": dataset_size,
            "split_type": split_type,
            "split_root_dir": str(split_root_dir),
        },
    }
    args["run"] = args_run
    logging.info(f"Saving the run parameters in {run_config_filepath}")
    yaml_write(to=run_config_filepath, data=args)
    logging.info(f"Running the training for {config['num_epochs']} epochs")
    train(trainer=trainer, num_epochs=config["num_epochs"])
    best_model_weights_output_dir = model_folder / "weights" / "best"
    save_best_model_weights(
        model_folder=model_folder,
        output_dir=best_model_weights_output_dir,
    )

    package_filepath = get_best_model_filepath(train_run_root_dir=record_path)
    logging.info(f"Packaging the model artifacts in one file in {package_filepath}")
    packaged = package_model(train_run_root_dir=record_path)
    torch.save(packaged, package_filepath)
