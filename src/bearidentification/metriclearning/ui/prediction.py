from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PIL import Image

from bearidentification.metriclearning.utils import IMAGENET_MEAN, IMAGENET_STD

DISTANCE_THRESHOLD_NEW_INDIVIDUAL = 0.7


def get_inverse_normalize_transform(mean, std):
    return torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def get_color(
    distance: float,
    distance_threshold_new_individual: float = DISTANCE_THRESHOLD_NEW_INDIVIDUAL,
    margin: float = 0.10,
) -> str:
    threshold_unsure = distance_threshold_new_individual * (1.0 - margin)
    threshold_new_individual = distance_threshold_new_individual * (1 + margin)
    if distance < threshold_unsure:
        return "green"
    elif distance < threshold_new_individual:
        return "orange"
    else:
        return "red"


def draw_extrated_chip(ax, chip_image) -> None:
    ax.set_title("Extracted chip")
    ax.set_axis_off()
    ax.imshow(chip_image)


def draw_closest_neighbors(
    fig: Figure,
    gs: GridSpec,
    i_start: int,
    k_closest_neighbors: int,
    indexed_k_nearest_individuals: dict,
) -> None:
    inv_normalize = get_inverse_normalize_transform(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    neighbors = []
    for bear_id, xs in indexed_k_nearest_individuals.items():
        for x in xs:
            data = x.copy()
            data["bear_id"] = bear_id
            neighbors.append(data)

    nearest_neighbors = sorted(
        neighbors,
        key=lambda x: x["distance"],
    )[:k_closest_neighbors]
    for j, neighbor in enumerate(nearest_neighbors):
        ax = fig.add_subplot(gs[i_start, j])
        distance = neighbor["distance"]
        bear_id = neighbor["bear_id"]
        dataset_image = neighbor["dataset_image"]
        image = inv_normalize(dataset_image).numpy()
        image = np.transpose(image, (1, 2, 0))
        color = get_color(distance=distance)
        ax.set_axis_off()
        ax.set_title(label=f"{bear_id}: {distance:.2f}", color=color)
        ax.imshow(image)


def draw_top_k_individuals(
    fig: Figure,
    gs: GridSpec,
    i_start: int,
    i_end: int,
    indexed_k_nearest_individuals: dict,
    bear_ids: list[str],
    indexed_samples: dict,
):
    inv_normalize = get_inverse_normalize_transform(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    for i in range(i_start, i_end):
        for j in range(len(bear_ids)):
            # Draw the closest individual chips
            if i == i_start:
                ax = fig.add_subplot(gs[i, j])
                bear_id = bear_ids[j]
                nearest_individual = indexed_k_nearest_individuals[bear_id][0]
                distance = nearest_individual["distance"]
                dataset_image = nearest_individual["dataset_image"]
                image = inv_normalize(dataset_image).numpy()
                image = np.transpose(image, (1, 2, 0))
                color = get_color(distance=distance)
                ax.set_axis_off()
                ax.set_title(label=f"{bear_id}: {distance:.2f}", color=color)
                ax.imshow(image)

            # Draw random chips from the same individuals
            else:
                bear_id = bear_ids[j]
                idx = i - i_start - 1
                if idx < len(indexed_samples[bear_id]):
                    filepath = indexed_samples[bear_id][idx]
                    if filepath:
                        ax = fig.add_subplot(gs[i, j])
                        with Image.open(filepath) as image:
                            ax.set_axis_off()
                            ax.imshow(image)


def bearid_ui(
    chip_filepath: Path,
    indexed_k_nearest_individuals: dict,
    indexed_samples: dict,
    save_filepath: Path,
    k_closest_neighbors: int = 5,
) -> None:
    """Main UI for identifying bears."""
    chip_image = Image.open(chip_filepath)
    # Assumption: the bear_ids are sorted by distance - if that's not something
    # we can rely on, we should just sort
    bear_ids = list(indexed_k_nearest_individuals.keys())

    # Max of the number of closest_neighbors and the number of bearids
    ncols = max(len(bear_ids), k_closest_neighbors)

    # 1 row for the extracted chip
    # 1 row for the closest neighbors title section
    # 1 row for the closest neighbors
    # 1 row for the individuals title section
    # rows for the indexed_samples (radom images of a given individual)
    nrows = max([len(xs) for xs in indexed_samples.values()]) + 4
    figsize = (3 * ncols, 3 * nrows)
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    font_properties_section = font_manager.FontProperties(size=35)
    font_properties_title = font_manager.FontProperties(size=40)

    # Draw the extracted chip
    i_chip = 0
    ax = fig.add_subplot(gs[i_chip, 0])
    draw_extrated_chip(ax, chip_image)

    # Draw closest neighbors
    i_closest_neighbors = 2
    ax = fig.add_subplot(gs[i_closest_neighbors - 1, :])
    ax.set_axis_off()
    ax.text(
        y=0.2,
        x=0,
        s="Closest faces",
        font_properties=font_properties_section,
    )
    draw_closest_neighbors(
        fig=fig,
        gs=gs,
        i_start=i_closest_neighbors,
        k_closest_neighbors=k_closest_neighbors,
        indexed_k_nearest_individuals=indexed_k_nearest_individuals,
    )
    # Filling out the grid with top k individuals and random samples
    i_top_k_individual = 4
    ax = fig.add_subplot(gs[i_top_k_individual - 1, :])
    ax.set_axis_off()
    ax.text(
        y=0.2,
        x=0,
        s=f"Closest {len(bear_ids)} individuals",
        font_properties=font_properties_section,
    )
    draw_top_k_individuals(
        fig=fig,
        gs=gs,
        i_end=nrows,
        i_start=i_top_k_individual,
        indexed_k_nearest_individuals=indexed_k_nearest_individuals,
        bear_ids=bear_ids,
        indexed_samples=indexed_samples,
    )

    fig.suptitle("BearID-v2", size=40)
    plt.savefig(save_filepath, bbox_inches="tight")
    plt.close()
