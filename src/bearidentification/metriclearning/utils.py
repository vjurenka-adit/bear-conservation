import logging
import os
import random
from pathlib import Path
from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_metric_learning.utils.logging_presets as logging_presets
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import umap
import yaml
from cycler import cycler
from PIL import Image
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

from bearidentification.metriclearning.embedder import MLP


def get_best_device() -> torch.device:
    """Returns the best torch device depending on the hardware it is running
    on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDumper(yaml.Dumper):
    """Formatter for dumping yaml."""

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def yaml_read(path: Path) -> dict:
    """Returns yaml content as a python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def yaml_write(to: Path, data: dict, dumper=MyDumper) -> None:
    """Writes a `data` dictionnary to the provided `to` path."""
    with open(to, "w") as f:
        yaml.dump(
            data=data,
            stream=f,
            Dumper=dumper,
            default_flow_style=False,
            sort_keys=False,
        )


# TODO
def validate_run_config(config: dict) -> bool:
    return True


def load_weights(
    network: torch.nn.Module,
    weights_filepath: Path,
    prefix: str = "",
) -> torch.nn.Module:
    """Loads the network weights.

    Returns the network.
    """
    assert weights_filepath.exists(), f"Invalid model_filepath {weights_filepath}"
    weights = torch.load(weights_filepath)
    prefixed_weights = prefix_keys_with(weights, prefix=prefix)
    network.load_state_dict(state_dict=prefixed_weights)
    return network


def prefix_keys_with(weights: OrderedDict, prefix: str = "module.") -> OrderedDict:
    """Returns the new weights where each key is prefixed with the provided
    `prefix`.

    Note: Useful when using DataParallel to account for the module. prefix key.
    """
    weights_copy = weights.copy()
    for k, v in weights.items():
        weights_copy[f"{prefix}{k}"] = v
        del weights_copy[k]
    return weights_copy


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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(transform_type: str = "bare", config: dict = {}) -> dict:
    """Returns a dict containing the transforms for the following splits:
    train, val, test and viz (the latter is used for batch visualization).

    transform_type should be in {bare, augmented}"""
    # FIXME: this is related to the model not to imageNET - should come from the config
    imagenet_crop_size = (224, 224)

    # Use to persist a batch of data as an artefact
    transform_viz = transforms.Compose(
        [
            transforms.Resize(imagenet_crop_size),
            transforms.ToTensor(),
        ]
    )

    transform_plain = transforms.Compose(
        [
            transforms.Resize(imagenet_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
                ),  # Taken from Dolphin ID
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
            record_path / "embeddings" / split_name / f"umap_epoch_{epoch}.png"
        )
        os.makedirs(output_filepath.parent, exist_ok=True)

        plt.savefig(output_filepath)
        plt.close()

    return visualizer_hook


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


def resize_dataframe(df: pd.DataFrame, threshold_value: int):
    return df.groupby("bear_id").filter(lambda x: len(x) > threshold_value)


def fix_random_seed(random_seed: int = 0) -> None:
    """Fix the random seed across dependencies."""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


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


def make_hooks(record_path: Path, experiment_name: str):
    """Creates the hooks for the training pipeline."""
    record_keeper, _, _ = logging_presets.get_record_keeper(
        csv_folder=record_path / "training_logs",
        tensorboard_folder=record_path / "tensorboard",
        experiment_name=experiment_name,
    )
    return logging_presets.get_hook_container(record_keeper=record_keeper)
