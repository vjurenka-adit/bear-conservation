import logging
import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import umap
from cycler import cycler
from PIL import Image
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

from bearidentification.metriclearning.embedder import MLP
from bearidentification.metriclearning.utils import yaml_write


def load_datasplit(
    split_type: str,
    dataset_size: str,
    split_root_dir: Path,
) -> pd.DataFrame:
    """
    args:
    split_type in {by_individual by_provided_bearid}
    dataset_size in {nano, small, medium, large, xlarge, full}

    Returns a dataframe with the loaded datasplit
    """
    filepath = split_root_dir / split_type / dataset_size / "data_split.csv"
    return pd.read_csv(filepath, sep=";")


def make_id_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe that maps a bear label (eg.

    bf_755) to a unique natural number (eg. 0). The dataFrame contains
    two columns, namely id and label.
    """
    return pd.DataFrame(
        list(enumerate(df["bear_id"].unique())), columns=["id", "label"]
    )


class BearDataset(Dataset):
    def __init__(self, dataframe, id_mapping, transform=None):
        self.dataframe = dataframe
        self.id_mapping = id_mapping
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        image_path = sample.path
        bear_id = sample.bear_id

        id_value = self.id_mapping.loc[self.id_mapping["label"] == bear_id, "id"].iloc[
            0
        ]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, id_value


def get_transforms(transform_type: str = "bare", config: dict = {}) -> dict:
    """Returns a dict containing the transforms for the following splits:
    train, val, test and viz (the latter is used for batch visualization).

    transform_type should be in {bare, augmented}"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_crop_size = (224, 224)

    # Use to persist a batch of data as an artefact
    transform_viz = transforms.Compose([transforms.ToTensor()])
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    if transform_type == "bare":
        return {
            "viz": transform_viz,
            "train": transform_plain,
            "val": transform_plain,
            "test": transform_plain,
        }
    elif transform_type == "augmented":
        # TODO: make use of the config here
        transform_train = transforms.Compose(
            [
                transforms.Resize(imagenet_crop_size),
                transforms.ColorJitter(
                    hue=0.1, 
                    saturation=(0.9, 1.1),
                ),                              # Taken from Dolphin ID
                v2.RandomRotation(degrees=10),  # Taken from Dolphin ID
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        return {
            "viz": transform_viz,
            # Only the train transform contains the data augmentation
            "train": transform_train,
            "val": transform_plain,
            "test": transform_plain,
        }
    else:
        raise Exception(f"transform_type {transform_type} not implemented.")


def make_dataloaders(
    batch_size: int,
    df_split: pd.DataFrame,
    transforms: dict,
) -> dict:
    """Returns a dict with top level keys in {dataset and loader}.

    Each returns a dict with the train, val and test objects associated.
    """
    df_train = df_split[df_split["split"] == "train"]
    df_val = df_split[df_split["split"] == "val"]
    df_test = df_split[df_split["split"] == "test"]
    id_mapping = make_id_mapping(df=df_split)

    viz_dataset = BearDataset(
        df_train,
        id_mapping,
        transform=transforms["viz"],
    )
    viz_loader = DataLoader(
        viz_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    train_dataset = BearDataset(
        df_train,
        id_mapping,
        transform=transforms["train"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = BearDataset(
        df_val,
        id_mapping,
        transform=transforms["val"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
    )

    test_dataset = BearDataset(
        df_test,
        id_mapping,
        transform=transforms["test"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    return {
        "dataset": {
            "viz": viz_dataset,
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        },
        "loader": {
            "viz": viz_loader,
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
    }


def make_visualizer_hook(record_path: Path):
    """Returns a visualizer_hook that renders the embeddings using UMAP on each
    split."""

    def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
        epoch = args[0]
        logging.info(
            f"UMAP plot - {split_name} split and label set {keyname} - epoch {epoch}"
        )
        logging.info(f"args: {args}")

        label_set = np.unique(labels)
        num_classes = len(label_set)

        fig = plt.figure(figsize=(20, 15))
        ax = plt.gca()
        ax.set_title(
            f"UMAP plot - {split_name} split and label set {keyname} - epoch {epoch}"
        )

        ax.set_prop_cycle(
            cycler(
                "color",
                [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)],
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(
                umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=5
            )

        output_filepath = (
            record_path / "embeddings" / f"umap_{split_name}_epoch_{epoch}.png"
        )
        os.makedirs(output_filepath.parent, exist_ok=True)

        plt.savefig(output_filepath)
        plt.close()

    return visualizer_hook


def get_experiment_name(
    experiment_number: int,
    loss_type: str,
    dataset_size: str,
    split_type: str,
) -> str:
    """Returns an experiment name based on the passed parameters."""
    return f"experiment_{experiment_number}_loss_{loss_type}_size_{dataset_size}_split_{split_type}"


def make_hooks(record_path: Path, experiment_name: str):
    """Creates the hooks for the training pipeline."""
    record_keeper, _, _ = logging_presets.get_record_keeper(
        csv_folder=record_path / "training_logs",
        tensorboard_folder=record_path / "tensorboard",
        experiment_name=experiment_name,
    )
    return logging_presets.get_hook_container(record_keeper=record_keeper)


def get_record_path(experiment_name: str, output_dir: Path) -> Path:
    """Returns the path where all the artefacts will be logged."""
    return output_dir / experiment_name


def make_tester(
    hooks,
    record_path: Path,
    accuracy_calculator: AccuracyCalculator,
):
    """Returns a tester used to evaluate and visualize the embeddings."""
    visualizer = umap.UMAP()
    visualizer_hook = make_visualizer_hook(record_path=record_path)
    return testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=visualizer,
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=2,
        accuracy_calculator=accuracy_calculator,
    )


def make_model_dict(
    device: torch.device,
    pretrained_backbone: str = "resnet18",
    embedding_size: int = 128,
    hidden_layer_sizes: list[int] = [1024],
) -> dict[str, nn.Module]:
    """
    Returns a dict with the following keys:
    - embedder: nn.Module - embedder model, usually an MLP.
    - trunk: nn.Module - the backbone model, usually a pretrained model (like a ResNet).
    """
    if pretrained_backbone == "resnet18":
        trunk = torchvision.models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    else:
        raise Exception(f"Cannot make trunk with backbone {pretrained_backbone}")

    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))

    embedder = torch.nn.DataParallel(
        MLP([trunk_output_size, *hidden_layer_sizes, embedding_size]).to(device)
    )
    return {
        "trunk": trunk,
        "embedder": embedder,
    }


def make_optimizers(config: dict, trunk: nn.Module, embedder: nn.Module) -> dict:
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
    mining_funcs: dict,
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


def get_best_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_sample_batch(dataloader: DataLoader, to: Path) -> None:
    """Draws a sample from the dataloader and persists it as an image grid in
    path `to`."""
    sample = next(iter(dataloader))
    imgs, _ = sample
    # create a grid
    plt.figure(figsize=(30, 15))
    grid = torchvision.utils.make_grid(nrow=20, tensor=imgs)
    image = np.transpose(grid, axes=(1, 2, 0))
    plt.imshow(image)
    os.makedirs(to.parent, exist_ok=True)
    plt.savefig(to)
    plt.close()


def make_accuracy_calculator() -> AccuracyCalculator:
    """Returns an accuracy calculator used to evaluate the performance of the
    model."""
    return AccuracyCalculator(k="max_bin_count")


def fix_random_seed(random_seed: int = 0) -> None:
    """Fix the random seed across dependencies."""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def run(
    split_root_dir: Path,
    split_type: str,
    dataset_size: str,
    output_dir: Path,
    config: dict,
    experiment_number: int,
    random_seed: int = 0,
) -> None:
    """Main entrypoint to setup a training pipeline based on the provided
    config.

    See src/bearidentification/metriclearning/configs/ for configs
    examples.
    """
    fix_random_seed(random_seed)

    experiment_name = get_experiment_name(
        experiment_number=experiment_number,
        loss_type=config["loss"]["type"],
        dataset_size=dataset_size,
        split_type=split_type,
    )
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
        transform_type="bare" if not "data_augmentation" in config else "augmented",
        config=config["data_augmentation"] if "data_augmentation" in config else {},
    )
    logging.info(f"Loaded the transforms: {transforms}")

    dataloaders = make_dataloaders(
        batch_size=config["batch_size"],
        df_split=df_split,
        transforms=transforms,
    )

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

    # TODO: use test set instead of val set when not present (example: by_individual split)
    dataset_dict = {
        "train": dataloaders["dataset"]["train"],
        "val": dataloaders["dataset"]["val"],
    }

    model_folder = record_path / "model"
    end_of_epoch_hook = make_end_of_epoch_hook(
        hooks=hooks,
        tester=tester,
        dataset_dict=dataset_dict,
        model_folder=model_folder,
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
        "experiment_number": experiment_number,
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
