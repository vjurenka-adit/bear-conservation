import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, samplers, trainers

from bearidentification.metriclearning.metrics import make_accuracy_calculator
from bearidentification.metriclearning.utils import (
    BearDataset,
    fix_random_seed,
    get_best_device,
    get_transforms,
    load_datasplit,
    make_dataloaders,
    make_hooks,
    make_id_mapping,
    make_model_dict,
    make_tester,
    resize_dataframe,
    save_sample_batch,
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


def make_optimizers(
    config: dict,
    trunk: nn.Module,
    embedder: nn.Module,
) -> dict:
    """Returns a dictionnary and one key for the embedder and one for trunk.

    The currently supported optimizer types are {adam}.
    """
    embedder_config = config["embedder"]["config"]
    trunk_config = config["trunk"]["config"]

    if config["embedder"]["type"] == "adam" and config["trunk"]["type"] == "adam":
        embedder_optimizer = torch.optim.Adam(
            embedder.parameters(),
            lr=embedder_config["lr"],
            weight_decay=embedder_config["weight_decay"],
        )
        trunk_optimizer = torch.optim.Adam(
            trunk.parameters(),
            lr=trunk_config["lr"],
            weight_decay=trunk_config["weight_decay"],
        )

        return {
            "trunk_optimizer": trunk_optimizer,
            "embedder_optimizer": embedder_optimizer,
        }
    else:
        raise Exception(f"The only supported optimizer type is Adam for now.")


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
    embedder_weights_best_path = list(model_folder.glob("embedder_best*"))[0]
    trunk_weights_best_path = list(model_folder.glob("trunk_best*"))[0]
    logging.info(f"Copying best model weights in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(src=embedder_weights_best_path, dst=output_dir / "embedder.pth")
    shutil.copy(src=trunk_weights_best_path, dst=output_dir / "trunk.pth")


def run(
    split_root_dir: Path,
    split_type: str,
    dataset_size: str,
    output_dir: Path,
    config: dict,
    # experiment_number: int,
    experiment_name: str,
    random_seed: int = 0,
) -> None:
    """Main entrypoint to setup a training pipeline based on the provided
    config.

    See src/bearidentification/metriclearning/configs/ for configs
    examples.
    """
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

    # Making smaller versions of the datasets for visualizing the
    # embeddings
    THRESHOLDS = {
        "nano": 150,
        "small": 100,
        "medium": 50,
        "large": 10,
        "xlarge": 1,
        "full": 0,
    }
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

    optimizers = make_optimizers(
        config=config["optimizers"],
        trunk=model_dict["trunk"],
        embedder=model_dict["embedder"],
    )
    logging.info(f"Initialized the optimizers {optimizers}")

    loss_funcs = make_loss_functions(
        loss_type=config["loss"]["type"],
        config=config["loss"]["config"],
    )
    logging.info(f"Initialized the loss_funcs: {loss_funcs}")

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

    # TODO: use test set instead of val set when not present (example:
    # by_individual split)
    dataset_dict = {
        "train": dataloaders["dataset"]["train"],
        "train_small": dataloaders_small["dataset"]["train"],
        "val": dataloaders["dataset"]["val"],
        "val_small": dataloaders_small["dataset"]["val"],
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
    save_best_model_weights(
        model_folder=model_folder,
        output_dir=model_folder / "weights" / "best",
    )


## REPL
# reorganize weights artifacts
# model_folder = Path(
#     "data/06_models/bearidentification/metric_learning/baseline_circleloss_dumb_nano_by_provided_bearid/model"
# )
# model_folder.exists()

# best_weights_output_dir = Path(
#     "data/06_models/bearidentification/metric_learning/baseline_circleloss_dumb_nano_by_provided_bearid/model/weights/best/"
# )

# # Find best weights in model_folder
# embedder_weights_best_path = list(model_folder.glob("embedder_best*"))[0]
# trunk_weights_best_path = list(model_folder.glob("trunk_best*"))[0]

# trunk_weights_best_path

# os.makedirs(best_weights_output_dir, exist_ok=True)
# shutil.copy(
#     src=embedder_weights_best_path, dst=best_weights_output_dir / "embedder.pth"
# )
# shutil.copy(src=trunk_weights_best_path, dst=best_weights_output_dir / "trunk.pth")


# model

## REPL
# split_root_dir = Path("data/04_feature/bearidentification/bearid/split/")
# split_root_dir.exists()

# split_type = "by_provided_bearid"
# dataset_size = "nano"
# df_split = load_datasplit(split_type=split_type, dataset_size=dataset_size)
# df_split.head()

# transform = get_transform(transform_type="dumb", config={})
# type(transform)

# batch_size = 32

# dataloaders = make_dataloaders(
#     batch_size=batch_size, df_split=df_split, transform=transform
# )
# dataloaders

# experiment_number = 0
# loss_type = "circleloss"
# experiment_name = get_experiment_name(
#     experiment_number=experiment_number,
#     loss_type=loss_type,
#     dataset_size=dataset_size,
#     split_type=split_type,
# )
# experiment_name
# output_dir = Path("data/06_models/bearidentification/metric_learning/")
# output_dir.exists()
# record_path = get_record_path(experiment_name=experiment_name, output_dir=output_dir)
# record_path
# hooks = make_hooks(record_path=record_path)
# hooks

# accuracy_calculator = AccuracyCalculator(k="max_bin_count")
# tester = make_tester(
#     hooks=hooks, record_path=record_path, accuracy_calculator=accuracy_calculator
# )
# tester

# device = get_best_device()

# pretrained_backbone = "resnet18"
# embedding_size = 128
# hidden_layer_sizes = [1024]

# model_dict = make_model_dict(
#     device=device,
#     pretrained_backbone=pretrained_backbone,
#     embedding_size=embedding_size,
#     hidden_layer_sizes=hidden_layer_sizes,
# )
# model_dict

# config_optimizers = {
#     "embedder": {"lr": 0.001, "weight_decay": 1e-5},
#     "trunk": {"lr": 0.0001, "weight_decay": 1e-5},
# }
# optimizers = make_optimizers(
#     config=config_optimizers,
#     trunk=model_dict["trunk"],
#     embedder=model_dict["embedder"],
# )
# optimizers

# config_loss = {"m": 0.4, "gamma": 256}
# loss_funcs = make_loss_functions(loss_type=loss_type, config=config_loss)
# loss_funcs

# config_miner = {"pos_strategy": "semihard", "neg_strategy": "hard"}
# miner_type = "batcheasyhardminer"
# mining_funcs = make_mining_functions(miner_type=miner_type, config=config_miner)
# mining_funcs

# id_mapping = make_id_mapping(df=df_split)
# id_mapping.head()
# config_sampler = {"m": 4}

# sampler = make_sampler(
#     sampler_type="mperclass",
#     id_mapping=id_mapping,
#     train_dataset=dataloaders["dataset"]["train"],
#     config=config_sampler,
# )
# sampler

# dataset_dict = {
#     "train": dataloaders["dataset"]["train"],
#     "val": dataloaders["dataset"]["val"],
# }
# record_path

# model_folder = record_path / "model"
# model_folder
# end_of_epoch_hook = make_end_of_epoch_hook(
#     hooks=hooks,
#     tester=tester,
#     dataset_dict=dataset_dict,
#     model_folder=model_folder,
# )
# end_of_epoch_hook

# trainer = make_trainer(
#     model_dict=model_dict,
#     optimizers=optimizers,
#     batch_size=batch_size,
#     loss_funcs=loss_funcs,
#     train_dataset=dataloaders["dataset"]["train"],
#     mining_funcs=mining_funcs,
#     sampler=sampler,
#     end_of_iteration_hook=hooks.end_of_iteration_hook,
#     end_of_epoch_hook=end_of_epoch_hook,
# )
# trainer

# # train(trainer=trainer, num_epochs=1)

# # run(
# #     split_root_dir=split_root_dir,
# #     split_type=split_type,
# #     dataset_size=dataset_size,
# #     output_dir=output_dir,
# #     config=config,
# #     experiment_number=42,
# #     random_seed=0,
# # )


## REPL to persist the config file
# from pathlib import Path

# from bearidentification.metriclearning.utils import yaml_read, yaml_write

# filepath = Path("src/bearidentification/metriclearning/configs/baseline.yaml")
# filepath.exists()

# yaml_write(to=filepath, data=config)

# # Check that it worked
# d = yaml_read(filepath)
# d == config

# config = {
#     "batch_size": 32,
#     "num_epochs": 1,
#     "data_augmentation": {"type": "bare", "config": {}},
#     "model": {
#         "trunk": {"backbone": "resnet18"},
#         "embedder": {
#             "embedding_size": 128,
#             "hidden_layer_sizes": [1024],
#         },
#     },
#     "loss": {
#         "type": "circleloss",
#         "config": {
#             "m": 0.4,
#             "gamma": 256,
#         },
#     },
#     "sampler": {
#         "type": "mperclass",
#         "config": {
#             "m": 4,
#         },
#     },
#     "optimizers": {
#         "embedder": {
#             "type": "adam",
#             "config": {
#                 "lr": 0.001,
#                 "weight_decay": 1e-5,
#             },
#         },
#         "trunk": {
#             "type": "adam",
#             "config": {
#                 "lr": 0.0001,
#                 "weight_decay": 1e-5,
#             },
#         },
#     },
#     "miner": {
#         "type": "batcheasyhardminer",
#         "config": {
#             "pos_strategy": "semihard",
#             "neg_strategy": "hard",
#         },
#     },
# }
