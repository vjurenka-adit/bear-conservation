from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from matplotlib.gridspec import GridSpec
from PIL import Image

from bearidentification.metriclearning.utils import IMAGENET_MEAN, IMAGENET_STD


def get_inverse_normalize_transform(mean, std):
    return torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def get_color(
    distance: float,
    distance_threshold_new_individual: float,
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


def bearid_ui(
    chip_filepath: Path,
    indexed_k_nearest_individuals: dict,
    indexed_samples: dict,
    save_filepath: Path,
    distance_threshold_new_individual: float = 0.7,
) -> None:
    """Main UI for identifying bears."""
    inv_normalize = get_inverse_normalize_transform(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    chip_image = Image.open(chip_filepath)
    # TODO: sort the bear ids by minimum distance here
    bear_ids = list(indexed_k_nearest_individuals.keys())
    ncols = len(bear_ids)
    nrows = max([len(xs) for xs in indexed_samples.values()]) + 1
    fig = plt.figure(constrained_layout=True, figsize=(3 * ncols, 3 * nrows))
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f"Extracted chip")
    ax.set_axis_off()
    ax.imshow(chip_image)
    nrow_start = 1
    for i in range(nrow_start, nrows):
        for j in range(ncols):
            # Draw the closest individual chips
            if i == nrow_start:
                ax = fig.add_subplot(gs[i, j])
                bear_id = bear_ids[j]
                nearest_individual = indexed_k_nearest_individuals[bear_id][0]
                distance = nearest_individual["distance"]
                dataset_image = nearest_individual["dataset_image"]
                image = inv_normalize(dataset_image).numpy()
                image = np.transpose(image, (1, 2, 0))
                color = get_color(
                    distance=distance,
                    distance_threshold_new_individual=distance_threshold_new_individual,
                    margin=0.10,
                )
                ax.set_axis_off()
                ax.set_title(label=f"{bear_id}: {distance:.2f}", color=color)
                ax.imshow(image)
            # Draw random chips from the same individuals
            else:
                bear_id = bear_ids[j]
                if i - nrow_start < len(indexed_samples[bear_id]):
                    filepath = indexed_samples[bear_id][i - nrow_start]
                    if filepath:
                        ax = fig.add_subplot(gs[i, j])
                        with Image.open(filepath) as image:
                            ax.set_axis_off()
                            ax.imshow(image)

    fig.suptitle("BearID-v2")
    plt.savefig(save_filepath, bbox_inches="tight")
    plt.close()
    # plt.show()
