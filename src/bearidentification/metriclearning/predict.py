import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from joblib.logger import shutil
from PIL import Image
from pytorch_metric_learning.utils.common_functions import logging
from pytorch_metric_learning.utils.inference import InferenceModel
from torch.utils.data import Dataset

from bearidentification.metriclearning.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_best_device,
    get_transforms,
    load_datasplit,
    load_weights,
    make_dataloaders,
    make_id_mapping,
    make_model_dict,
    yaml_read,
)


def compute_knn(
    model: InferenceModel,
    dataset: Dataset,
    knn_index_filepath: Path,
) -> None:
    """Retrieves or computes the KNN embeddings of the model."""
    # Cache the KNN index
    if knn_index_filepath.exists():
        logging.info("Loading from cache")
        model.load_knn_func(filename=str(knn_index_filepath))
    else:
        logging.info("Training KNN")
        model.train_knn(dataset)
        os.makedirs(knn_index_filepath.parent, exist_ok=True)
        model.save_knn_func(filename=str(knn_index_filepath))


def get_inverse_normalize_transform(mean, std):
    return torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def save_nearest_images(
    nearest_images: list,
    bearids: list[str],
    distances: list[float],
    save_filepath: Path,
) -> None:
    inv_normalize = get_inverse_normalize_transform(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    k = len(nearest_images)
    n_row = 1
    n_col = k
    _, axs = plt.subplots(n_row, n_col, figsize=(3 * k, 12))
    axs = axs.flatten()

    for nearest_image, bearid, distance, ax in zip(
        nearest_images, bearids, distances, axs
    ):
        image = inv_normalize(nearest_image).numpy()
        image = np.transpose(image, (1, 2, 0))
        ax.set_axis_off()
        ax.set_title(f"{bearid}: {distance:.2f}")
        ax.imshow(image)

    plt.savefig(save_filepath, bbox_inches="tight")
    plt.close()


def make_id_to_label(id_mapping: pd.DataFrame) -> dict[int, str]:
    return id_mapping.set_index("id")["label"].to_dict()


def run(
    model_filepath: Path,
    k: int,
    knn_index_filepath: Path,
    chip_filepath: Path,
    output_dir: Path,
) -> None:
    """Main entrypoints to run the inference.

    args:
    - trunk_weights_filepath: filepath of the trunk weights
    - embedder_weights_filepath: filepath of the embedder weights
    - knn_index_filepath: filepath to cache the knn computed index
    - k: integer - k closest embeddings to return
    - output_dir: path where results are stored
    - args_filepath: arguments used to train the model (yaml file)
    - chip_path: path of the chip to run the identification on
    """
    print(f"model_filepath: {model_filepath}")
    loaded_model = torch.load(model_filepath)
    device = get_best_device()
    args = loaded_model["args"]
    config = args.copy()
    del config["run"]

    transforms = get_transforms(
        data_augmentation=config.get("data_augmentation", {}),
        trunk_preprocessing=config["model"]["trunk"].get("preprocessing", {}),
    )

    logging.info(f"loading the df_split")
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

    # TODO: embed the full dataset here?
    # TODO: get rid of the bursts
    dataset_train = dataloaders["dataset"]["train"]

    compute_knn(
        model=model,
        dataset=dataset_train,
        knn_index_filepath=knn_index_filepath,
    )

    logging.info(f"Image Path: {chip_filepath}")
    image = Image.open(chip_filepath)
    transform_test = transforms["test"]
    model_input = transform_test(image)
    id_to_label = make_id_to_label(id_mapping=id_mapping)
    distances, indices = model.get_nearest_neighbors(
        query=model_input.unsqueeze(0),
        k=k,
    )
    indices_on_cpu = indices.cpu()[0].tolist()
    nearest_imgs, nearest_ids = list(
        zip(*[dataset_train[idx] for idx in indices_on_cpu])
    )
    prediction_dir = output_dir
    os.makedirs(prediction_dir, exist_ok=True)
    shutil.copy(
        src=chip_filepath,
        dst=prediction_dir / "chip.jpg",
    )
    bearids = [id_to_label.get(nearest_id, "unknown") for nearest_id in nearest_ids]

    save_nearest_images(
        nearest_images=list(nearest_imgs),
        bearids=bearids,
        distances=torch.flatten(distances).tolist(),
        save_filepath=prediction_dir / f"prediction_at_{k}.png",
    )
