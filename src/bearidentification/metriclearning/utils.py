import logging
import os
import random
from pathlib import Path
from typing import Optional, OrderedDict

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

from bearidentification.metriclearning.model.embedder import MLP


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


def validate_run_config(config: dict) -> bool:
    keyset_config = set(config.keys())
    keyset_minimum = {
        "batch_size",
        "num_epochs",
        "patience",
        "model",
        "loss",
        "sampler",
        "optimizers",
        "miner",
    }
    keyset_maximum = keyset_minimum | {"data_augmentation"}
    if keyset_config < keyset_minimum:
        logging.error(f"Does not contain the required keys {keyset_minimum}")
        return False
    elif keyset_config > keyset_maximum:
        logging.error(f"Contains unsupported keys {keyset_maximum}")
        return False
    else:
        return True


def load_weights(
    network: torch.nn.Module,
    weights_filepath: Optional[Path] = None,
    weights: Optional[OrderedDict] = None,
    prefix: str = "",
) -> torch.nn.Module:
    """Loads the network weights.

    Returns the network.
    """
    if weights:
        prefixed_weights = prefix_keys_with(weights, prefix=prefix)
        network.load_state_dict(state_dict=prefixed_weights)
        return network
    elif weights_filepath:
        assert weights_filepath.exists(), f"Invalid model_filepath {weights_filepath}"
        weights = torch.load(weights_filepath)
        prefixed_weights = prefix_keys_with(weights, prefix=prefix)
        network.load_state_dict(state_dict=prefixed_weights)
        return network
    else:
        raise Exception(f"Should provide at least weights or weights_filepath")


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


# TODO: move to utils
def filter_none(xs: list) -> list:
    return [x for x in xs if x is not None]


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "int64":
        return torch.int64
    else:
        logging.warning(
            f"dtype_str {dtype_str} not implemented, returning default value"
        )
        return torch.float32


def get_transforms(
    data_augmentation: dict = {},
    trunk_preprocessing: dict = {},
) -> dict:
    """Returns a dict containing the transforms for the following splits:
    train, val, test and viz (the latter is used for batch visualization).
    """
    logging.info(f"data_augmentation config: {data_augmentation}")
    logging.info(f"trunk preprocessing config: {trunk_preprocessing}")

    DEFAULT_CROP_SIZE = 224
    crop_size = (
        trunk_preprocessing.get("crop_size", DEFAULT_CROP_SIZE),
        trunk_preprocessing.get("crop_size", DEFAULT_CROP_SIZE),
    )

    # transform to persist a batch of data as an artefact
    transform_viz = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.ToTensor(),
        ]
    )

    mdtype: Optional[torch.dtype] = (
        get_dtype(trunk_preprocessing["values"].get("dtype", None))
        if trunk_preprocessing.get("values", None)
        else None
    )
    mscale: Optional[bool] = (
        trunk_preprocessing["values"].get("scale", None)
        if trunk_preprocessing.get("values", None)
        else None
    )

    mmean: Optional[list[float]] = (
        trunk_preprocessing["normalization"].get("mean", None)
        if trunk_preprocessing.get("normalization", None)
        else None
    )

    mstd: Optional[list[float]] = (
        trunk_preprocessing["normalization"].get("std", None)
        if trunk_preprocessing.get("normalization", None)
        else None
    )

    hue = (
        data_augmentation["colorjitter"].get("hue", 0)
        if data_augmentation.get("colorjitter", 0)
        else 0
    )
    saturation = (
        data_augmentation["colorjitter"].get("saturation", 0)
        if data_augmentation.get("colorjitter", 0)
        else 0
    )
    degrees = (
        data_augmentation["rotation"].get("degrees", 0)
        if data_augmentation.get("rotation", 0)
        else 0
    )

    transformations_plain = [
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        v2.ToDtype(dtype=mdtype, scale=mscale) if mdtype and mscale else None,
        transforms.Normalize(mean=mmean, std=mstd) if mmean and mstd else None,
    ]

    transformations_train = [
        transforms.Resize(crop_size),
        transforms.ColorJitter(
            hue=hue,
            saturation=saturation,
        )
        if data_augmentation.get("colorjitter", None)
        else None,  # Taken from Dolphin ID
        v2.RandomRotation(degrees=degrees)
        if data_augmentation.get("rotation", None)
        else None,  # Taken from Dolphin ID
        transforms.ToTensor(),
        v2.ToDtype(dtype=mdtype, scale=mscale) if mdtype and mscale else None,
        transforms.Normalize(mean=mmean, std=mstd) if mmean and mstd else None,
    ]

    # Filtering out None transforms
    transform_plain = transforms.Compose(filter_none(transformations_plain))
    transform_train = transforms.Compose(filter_none(transformations_train))

    return {
        "viz": transform_viz,
        "train": transform_train,
        "val": transform_plain,
        "test": transform_plain,
    }


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
    full_dataset = BearDataset(
        df_split,
        id_mapping,
        transform=transforms["val"],
    )

    return {
        "dataset": {
            "viz": viz_dataset,
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "full": full_dataset,
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


def check_backbone(pretrained_backbone: str) -> None:
    allowed_backbones = {
        "resnet18",
        "resnet50",
        "convnext_tiny",
        "convnext_base",
        "convnext_large",
        "efficientnet_v2_s",
        # "squeezenet1_1",
        "vit_b_16",
    }
    assert (
        pretrained_backbone in allowed_backbones
    ), f"pretrained_backbone {pretrained_backbone} is not implemented, only {allowed_backbones}"


def make_trunk(pretrained_backbone: str = "resnet18") -> nn.Module:
    """Returns a nn.Module with pretrained weights using a given
    pretrained_backbone.

    Note: The currently available backbones are resnet18, resnet50,
    convnext_tiny, convnext_bas, efficientnet_v2_s, squeezenet1_1, vit_b_16
    """

    check_backbone(pretrained_backbone)

    if pretrained_backbone == "resnet18":
        return torchvision.models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "resnet50":
        return torchvision.models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "convnext_tiny":
        return torchvision.models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "convnext_base":
        return torchvision.models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "convnext_large":
        return torchvision.models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "efficientnet_v2_s":
        return torchvision.models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "squeezenet1_1":
        return torchvision.models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        )
    elif pretrained_backbone == "vit_b_16":
        return torchvision.models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        )
    else:
        raise Exception(f"Cannot make trunk with backbone {pretrained_backbone}")


def make_embedder(
    pretrained_backbone: str,
    trunk: nn.Module,
    embedding_size: int,
    hidden_layer_sizes: list[int],
) -> nn.Module:
    check_backbone(pretrained_backbone)

    if pretrained_backbone in ["resnet18", "resnet50"]:
        trunk_output_size = trunk.fc.in_features
        trunk.fc = nn.Identity()
        return MLP([trunk_output_size, *hidden_layer_sizes, embedding_size])
    elif pretrained_backbone in ["convnext_tiny", "convnext_base", "convnext_large"]:
        trunk_output_size = trunk.classifier[-1].in_features
        trunk.classifier[-1] = nn.Identity()
        return MLP([trunk_output_size, *hidden_layer_sizes, embedding_size])
    elif pretrained_backbone == "efficientnet_v2_s":
        trunk_output_size = trunk.classifier[-1].in_features
        trunk.classifier[-1] = nn.Identity()
        return MLP([trunk_output_size, *hidden_layer_sizes, embedding_size])
    elif pretrained_backbone == "vit_b_16":
        trunk_output_size = trunk.heads.head.in_features
        trunk.heads.head = nn.Identity()
        return MLP([trunk_output_size, *hidden_layer_sizes, embedding_size])
    else:
        raise Exception(f"{pretrained_backbone} embedder not implemented yet")


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

    trunk = make_trunk(pretrained_backbone=pretrained_backbone)
    embedder = make_embedder(
        pretrained_backbone=pretrained_backbone,
        embedding_size=embedding_size,
        hidden_layer_sizes=hidden_layer_sizes,
        trunk=trunk,
    )

    trunk = torch.nn.DataParallel(trunk.to(device))
    embedder = torch.nn.DataParallel(embedder.to(device))

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


def make_hooks(
    record_path: Path,
    experiment_name: str,
    primary_metric: str = "precision_at_1",
):
    """Creates the hooks for the training pipeline."""
    record_keeper, _, _ = logging_presets.get_record_keeper(
        csv_folder=record_path / "training_logs",
        tensorboard_folder=record_path / "tensorboard",
        experiment_name=experiment_name,
    )
    return logging_presets.get_hook_container(
        record_keeper=record_keeper,
        primary_metric=primary_metric,
    )


def get_best_weights_filepaths(model_folder: Path) -> dict:
    """Returns the best weights filepaths for the trunk and the embedder."""
    embedder_weights_best_path = list(model_folder.glob("embedder_best*"))[0]
    trunk_weights_best_path = list(model_folder.glob("trunk_best*"))[0]
    return {
        "trunk": trunk_weights_best_path,
        "embedder": embedder_weights_best_path,
    }


def get_best_weights(model_folder: Path) -> dict:
    """Returns the best weights for the trunk and the embedder."""
    best_weights_filepaths = get_best_weights_filepaths(model_folder=model_folder)
    return {
        "trunk": torch.load(best_weights_filepaths["trunk"]),
        "embedder": torch.load(best_weights_filepaths["embedder"]),
    }


def get_best_model_filepath(train_run_root_dir: Path) -> Path:
    """Returns the best model filepath."""
    return train_run_root_dir / "model" / "weights" / "best" / "model.pth"
