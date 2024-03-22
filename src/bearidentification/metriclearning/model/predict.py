import os
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from joblib.logger import shutil
from matplotlib.gridspec import GridSpec
from PIL import Image
from pytorch_metric_learning.utils.common_functions import logging
from pytorch_metric_learning.utils.inference import InferenceModel
from torch.utils.data import Dataset

from bearidentification.metriclearning.ui.prediction import bearid_ui
from bearidentification.metriclearning.utils import (
    get_best_device,
    get_transforms,
    load_weights,
    make_dataloaders,
    make_id_mapping,
    make_model_dict,
)


def _aux_get_k_nearest_individuals(
    model: InferenceModel,
    k_neighbors: int,
    k_individuals: int,
    query,
    id_to_label: dict,
    dataset: Dataset,
) -> dict:
    """Auxiliary helper function to get k nearest individuals.

    Returns a dict with the following keys:
    - k_neighbors: int - number of neighbors the KNN search extends to in order to find at least k_individuals
    - dataset_indices: list[int] - list of indices to call get_item on the dataset
    - dataset_labels: list[int] - labels of the dataset for the given dataset_indices
    - dataset_images: list[torch.tensor] - chips of the bears
    - distances: list[float] - distances from the query

    Note: it can return more than k_individuals as it extends progressively the
    KNN search to find at least k_individuals.
    """
    assert k_individuals <= 20, f"Keep a small k_individuals: {k_individuals}"

    distances, indices = model.get_nearest_neighbors(query=query, k=k_neighbors)
    indices_on_cpu = indices.cpu()[0].tolist()
    distances_on_cpu = distances.cpu()[0].tolist()
    nearest_images, nearest_ids = list(zip(*[dataset[idx] for idx in indices_on_cpu]))
    bearids = [id_to_label.get(nearest_id, "unknown") for nearest_id in nearest_ids]
    counter = Counter(nearest_ids)
    if len(counter.keys()) >= k_individuals:
        return {
            "k_neighbors": k_neighbors,
            "dataset_indices": indices_on_cpu,
            "dataset_labels": list(nearest_ids),
            "dataset_images": list(nearest_images),
            "bearids": bearids,
            "distances": distances_on_cpu,
        }
    else:
        new_k_neighbors = k_neighbors * 2
        return _aux_get_k_nearest_individuals(
            model,
            k_neighbors=new_k_neighbors,
            k_individuals=k_individuals,
            query=query,
            id_to_label=id_to_label,
            dataset=dataset,
        )


def _find_cutoff_index(k: int, dataset_labels: list[str]) -> Optional[int]:
    """Returns the index for dataset_labels that retrieves exactly k
    individuals."""
    if not dataset_labels:
        return None
    else:
        selected_labels = set()
        cutoff_index = -1
        for idx, label in enumerate(dataset_labels):
            if len(selected_labels) == k:
                break
            else:
                selected_labels.add(label)
                cutoff_index = idx + 1
        return cutoff_index


def get_k_nearest_individuals(
    model: InferenceModel,
    k: int,
    query,
    id_to_label: dict,
    dataset: Dataset,
) -> dict:
    """Returns the k nearest individuals using the inference model and a query.

    A dict is returned with the following keys:
    - dataset_indices: list[int] - list of indices to call get_item on the dataset
    - dataset_labels: list[int] - labels of the dataset for the given dataset_indices
    - dataset_images: list[torch.tensor] - chips of the bears
    - distances: list[float] - distances from the query
    """
    k_neighbors = k * 5
    k_individuals = k
    result = _aux_get_k_nearest_individuals(
        model=model,
        k_neighbors=k_neighbors,
        k_individuals=k_individuals,
        query=query,
        id_to_label=id_to_label,
        dataset=dataset,
    )
    cutoff_index = _find_cutoff_index(
        k=k,
        dataset_labels=result["dataset_labels"],
    )
    return {
        "dataset_indices": result["dataset_indices"][:cutoff_index],
        "dataset_labels": result["dataset_labels"][:cutoff_index],
        "dataset_images": result["dataset_images"][:cutoff_index],
        "bearids": result["bearids"][:cutoff_index],
        "distances": result["distances"][:cutoff_index],
    }


def index_by_bearid(k_nearest_individuals: dict) -> dict:
    """Returns a dict where keys are bearid labels (eg. 'bf_480') and the
    values are list of the following dict shapes:

    - dataset_label: int
    - dataset_image: torch.tensor
    - distance: float
    - dataset_index: int
    """
    result = {}
    for dataset_label, dataset_image, distance, bearid, dataset_index in zip(
        k_nearest_individuals["dataset_labels"],
        k_nearest_individuals["dataset_images"],
        k_nearest_individuals["distances"],
        k_nearest_individuals["bearids"],
        k_nearest_individuals["dataset_indices"],
    ):
        row = {
            "dataset_label": dataset_label,
            "dataset_image": dataset_image,
            "distance": distance,
            "dataset_index": dataset_index,
        }
        if bearid not in result:
            result[bearid] = [row]
        else:
            result[bearid].append(row)
    return result


def sample_chips_from_bearid(
    bear_id: str,
    df_split: pd.DataFrame,
    n: int = 4,
) -> list[Path]:
    xs = df_split[df_split["bear_id"] == bear_id].sample(n=n)["path"].tolist()
    return [Path(x) for x in xs]


def make_indexed_samples(
    bear_ids: list[str],
    df_split: pd.DataFrame,
    n: int = 4,
) -> dict[str, list[Path]]:
    return {
        bear_id: sample_chips_from_bearid(bear_id=bear_id, df_split=df_split, n=n)
        for bear_id in bear_ids
    }


# def compute_knn(
#     model: InferenceModel,
#     dataset: Dataset,
#     knn_index_filepath: Path,
# ) -> None:
#     """Retrieves or computes the KNN embeddings of the model."""
#     # Cache the KNN index
#     if knn_index_filepath.exists():
#         logging.info("Loading from cache")
#         model.load_knn_func(filename=str(knn_index_filepath))
#     else:
#         logging.info("Training KNN")
#         model.train_knn(dataset)
#         os.makedirs(knn_index_filepath.parent, exist_ok=True)
#         model.save_knn_func(filename=str(knn_index_filepath))


def make_id_to_label(id_mapping: pd.DataFrame) -> dict[int, str]:
    return id_mapping.set_index("id")["label"].to_dict()


def run(
    model_filepath: Path,
    k: int,
    knn_index_filepath: Path,
    chip_filepath: Path,
    output_dir: Path,
    n_samples_per_individual: int = 5,
) -> None:
    """Main entrypoints to run the inference.

    args:
    - model_filepath: filepath of the metriclearning model
    - knn_index_filepath: filepath to cache the knn computed index
    - k: integer - k closest individuals to return
    - output_dir: path where results are stored
    - chip_filepath: path of the chip to run the identification on
    - n_samples_per_individual: integer - number of samples per individuals to retrieve
    """
    device = get_best_device()
    loaded_model = torch.load(model_filepath, map_location=device)
    args = loaded_model["args"]
    config = args.copy()
    del config["run"]

    transforms = get_transforms(
        data_augmentation=config.get("data_augmentation", {}),
        trunk_preprocessing=config["model"]["trunk"].get("preprocessing", {}),
    )

    logging.info("loading the df_split")
    df_split = pd.DataFrame(loaded_model["data_split"])
    df_split.info()

    id_mapping = make_id_mapping(df=df_split)

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

    model = InferenceModel(
        trunk=trunk,
        embedder=embedder,
    )

    dataset_full = dataloaders["dataset"]["full"]

    assert (
        knn_index_filepath.exists()
    ), f"knn_index_filepath invalid filepath: {knn_index_filepath}"
    model.load_knn_func(filename=str(knn_index_filepath))

    logging.info(f"Image Path: {chip_filepath}")
    image = Image.open(chip_filepath)
    transform_test = transforms["test"]
    model_input = transform_test(image)
    query = model_input.unsqueeze(0)
    id_to_label = make_id_to_label(id_mapping=id_mapping)
    k_nearest_individuals = get_k_nearest_individuals(
        model=model,
        k=k,
        query=query,
        id_to_label=id_to_label,
        dataset=dataset_full,
    )
    indexed_k_nearest_individuals = index_by_bearid(
        k_nearest_individuals=k_nearest_individuals
    )
    bear_ids = list(indexed_k_nearest_individuals.keys())
    indexed_samples = make_indexed_samples(
        bear_ids=bear_ids,
        df_split=df_split,
        n=n_samples_per_individual,
    )
    prediction_dir = output_dir
    os.makedirs(prediction_dir, exist_ok=True)
    shutil.copy(
        src=chip_filepath,
        dst=prediction_dir / "chip.jpg",
    )
    bearid_ui(
        chip_filepath=chip_filepath,
        indexed_k_nearest_individuals=indexed_k_nearest_individuals,
        indexed_samples=indexed_samples,
        save_filepath=Path(
            prediction_dir
            / f"prediction_at_{k}_individuals_{n_samples_per_individual}_samples_per_individual.png"
        ),
    )
