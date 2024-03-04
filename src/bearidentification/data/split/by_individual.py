import logging
import os
import random
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from bearidentification.data.split.utils import (
    ALLOWED_ORIGINS,
    THRESHOLDS,
    MyDumper,
    resize_dataframe,
)


def list_subdirs(dir: Path):
    """Lists all subdirs in dir."""
    return [item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item))]


def collect_samples(chips_root_dir: Path, allowed_origins: list[str]) -> list[dict]:
    """Locate every chip in the dataset and returns a list of dicts
    representing the collected chips."""
    samples = []

    for origin in allowed_origins:
        for encounter in list_subdirs(chips_root_dir / origin):
            for bear_id in list_subdirs(chips_root_dir / origin / encounter):
                for image in os.listdir(chips_root_dir / origin / encounter / bear_id):
                    image_path = chips_root_dir / origin / encounter / bear_id / image
                    samples.append(
                        {
                            "origin": origin,
                            "encounter": encounter,
                            "bear_id": bear_id,
                            "image": image,
                            "path": image_path,
                        }
                    )
    return samples


def filter_dataframe(df: pd.DataFrame) -> dict:
    """Filters the dataframe based on rules.

    Keep only rows where there are more than 1 bear images.
    """
    df_filtered = df.groupby("bear_id").filter(lambda x: len(x) > 1)
    logging.info(f"Discarded {len(df) - len(df_filtered)} rows")
    df_one_individual = df.groupby("bear_id").filter(lambda x: len(x) == 1)

    return {
        "df_filtered": df_filtered,
        "df_one_individual": df_one_individual,
    }


def train_test_split(
    df_raw: pd.DataFrame,
    train_size_ratio: float = 0.7,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Adds a `split` column with values in {train, test} depending on the
    train_size_ratio and the random_seed.

    Bears with more than one encounter are split into train and test sets according to the `train_size_ratio`.
    All the bears with only one encounter are placed in the test set.
    """

    d = filter_dataframe(df_raw)
    df_filtered, df_one_individual = d["df_filtered"], d["df_one_individual"]

    bears = [x for _, x in df_filtered.groupby("bear_id")]
    single_bears = [x for _, x in df_one_individual.groupby("bear_id")]

    shuffled_bears = random.Random(random_seed).sample(bears, len(bears))
    train_split = shuffled_bears[: int(train_size_ratio * len(shuffled_bears))]
    test_split = shuffled_bears[int(train_size_ratio * len(shuffled_bears)) :]
    test_split.extend(
        single_bears
    )  # We can use the individuals with one image during testing
    df_train = pd.concat(train_split)
    df_train["split"] = "train"
    df_test = pd.concat(test_split)
    df_test["split"] = "test"

    return pd.concat([df_train, df_test])


def write_config_yaml(
    path: Path,
    df: pd.DataFrame,
    train_size_ratio: float,
    random_seed: int,
    allowed_origins: list[str],
    chips_root_dir: Path,
    threshold_value: int,
) -> None:
    """Writes the `config.yaml` file that describes the generated datasplit."""

    data = {
        "train_size_ratio": train_size_ratio,
        "train_dataset_size": len(df[df["split"] == "train"]),
        "test_dataset_size": len(df[df["split"] == "test"]),
        "train_dataset_number_individuals": len(
            df[df["split"] == "train"].bear_id.unique()
        ),
        "test_dataset_number_individuals": len(
            df[df["split"] == "test"].bear_id.unique()
        ),
        "random_seed": random_seed,
        "allowed_origins": allowed_origins,
        "chips_root_dir": str(chips_root_dir),
        "threshold_value": threshold_value,
    }

    with open(path / "config.yaml", "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def build_datasplit(
    chips_root_dir: Path,
    allowed_origins: list[str],
    threshold_value: int,
    random_seed: int = 0,
    train_size_ratio: float = 0.7,
) -> pd.DataFrame:
    samples = collect_samples(
        chips_root_dir=chips_root_dir,
        allowed_origins=allowed_origins,
    )
    df_raw = pd.DataFrame(samples)
    df_resized = resize_dataframe(df_raw, threshold_value=threshold_value)

    df = train_test_split(
        df_raw=df_resized,
        train_size_ratio=train_size_ratio,
        random_seed=random_seed,
    )

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    # Sanity check
    assert len(df_resized.bear_id.unique()) == len(df_train.bear_id.unique()) + len(
        df_test.bear_id.unique()
    )

    return df


def save_datasplit(
    df: pd.DataFrame,
    save_dir: Path,
    chips_root_dir: Path,
    threshold_value: int,
    allowed_origins: list[str],
    random_seed: int = 0,
    train_size_ratio: float = 0.7,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    output_filepath = save_dir / "data_split.csv"
    logging.info(f"Saving split in {output_filepath}")
    df.to_csv(output_filepath, sep=";", index=False)
    write_config_yaml(
        path=save_dir,
        df=df,
        train_size_ratio=train_size_ratio,
        random_seed=random_seed,
        allowed_origins=allowed_origins,
        chips_root_dir=chips_root_dir,
        threshold_value=threshold_value,
    )


def save_all_datasplits(
    chips_root_dir: Path,
    save_dir: Path,
    thresholds: dict,
    allowed_origins: list[str],
    random_seed: int,
    train_size_ratio: float,
) -> None:
    for threshold_key in tqdm(thresholds.keys()):
        logging.info(f"generating datasplit for key: {threshold_key}")
        threshold_value = thresholds[threshold_key]
        df = build_datasplit(
            chips_root_dir=chips_root_dir,
            allowed_origins=allowed_origins,
            threshold_value=threshold_value,
            random_seed=random_seed,
            train_size_ratio=train_size_ratio,
        )
        output_dir = save_dir / f"by_individual/{threshold_key}"
        # save_dir = Path(
        #     f"../../../data/04_feature/bearidentification/bearid/data_split/by_individual/{threshold_key}/"
        # )
        save_datasplit(
            df=df,
            save_dir=output_dir,
            chips_root_dir=chips_root_dir,
            threshold_value=threshold_value,
            allowed_origins=allowed_origins,
            random_seed=random_seed,
            train_size_ratio=train_size_ratio,
        )


def run(
    chips_root_dir: Path,
    save_dir: Path,
    thresholds: dict = THRESHOLDS,
    allowed_origins: list[str] = ALLOWED_ORIGINS,
    random_seed: int = 0,
    train_size_ratio: float = 0.7,
) -> None:
    return save_all_datasplits(
        chips_root_dir=chips_root_dir,
        save_dir=save_dir,
        thresholds=thresholds,
        allowed_origins=allowed_origins,
        random_seed=random_seed,
        train_size_ratio=train_size_ratio,
    )
