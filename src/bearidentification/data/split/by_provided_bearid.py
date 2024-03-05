import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from bearidentification.data.split.utils import THRESHOLDS, MyDumper, resize_dataframe


def parse_xml_chips(filepath: Path, chips_root_dir: Path) -> pd.DataFrame:
    results = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    element_chips = root.find("chips")
    if element_chips:
        for chip in element_chips.findall("chip"):
            file_attribute = chip.get("file")
            if file_attribute:
                _, origin, encounter, label_id, chip_filename = file_attribute.split(
                    "/"
                )
                new_chip_filename = chip_filename.replace("_chip_0", "")
                chip_filepath = (
                    chips_root_dir / origin / encounter / label_id / new_chip_filename
                )

                results.append(
                    {
                        "origin": origin,
                        "encounter": encounter,
                        "bear_id": label_id,
                        "image": chip_filename,
                        "path": str(chip_filepath),
                        "path_exists": chip_filepath.exists(),
                    }
                )
    return pd.DataFrame(results)


def build_datasplit(
    bearid_root_path: Path,
    chips_root_dir: Path,
) -> pd.DataFrame:
    df_train = parse_xml_chips(
        filepath=bearid_root_path / "chips_train.xml", chips_root_dir=chips_root_dir
    )
    df_val = parse_xml_chips(
        filepath=bearid_root_path / "chips_val.xml", chips_root_dir=chips_root_dir
    )
    df_test = parse_xml_chips(
        filepath=bearid_root_path / "chips_test.xml", chips_root_dir=chips_root_dir
    )
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_split = pd.concat([df_train, df_val, df_test])
    df_dropped_split = df_split.drop(df_split[~df_split["path_exists"]].index)
    return df_dropped_split.drop(columns=["path_exists"])


def write_config_yaml(
    path: Path,
    df: pd.DataFrame,
    chips_root_dir: Path,
    threshold_value: int,
) -> None:
    """Writes the `config.yaml` file that describes the generated datasplit."""

    data = {
        "train_dataset_size": len(df[df["split"] == "train"]),
        "val_dataset_size": len(df[df["split"] == "val"]),
        "test_dataset_size": len(df[df["split"] == "test"]),
        "train_dataset_number_individuals": len(
            df[df["split"] == "train"].bear_id.unique()
        ),
        "val_dataset_number_individuals": len(
            df[df["split"] == "val"].bear_id.unique()
        ),
        "test_dataset_number_individuals": len(
            df[df["split"] == "test"].bear_id.unique()
        ),
        "chips_root_dir": str(chips_root_dir),
        "threshold_value": threshold_value,
    }

    with open(path / "config.yaml", "w") as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def save_datasplit(
    df: pd.DataFrame,
    chips_root_dir: Path,
    threshold_value: int,
    save_dir: Path,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    output_filepath = save_dir / "data_split.csv"
    logging.info(f"Saving split in {output_filepath}")
    df.to_csv(output_filepath, sep=";", index=False)
    write_config_yaml(
        path=save_dir,
        df=df,
        chips_root_dir=chips_root_dir,
        threshold_value=threshold_value,
    )


def save_all_datasplits(
    chips_root_dir: Path,
    bearid_root_path: Path,
    save_dir: Path,
    thresholds: dict,
) -> None:
    df = build_datasplit(
        bearid_root_path=bearid_root_path,
        chips_root_dir=chips_root_dir,
    )

    for threshold_key in tqdm(thresholds.keys()):
        logging.info(f"generating datasplit for key: {threshold_key}")
        threshold_value = thresholds[threshold_key]
        df_resized = resize_dataframe(df=df, threshold_value=threshold_value)
        output_dir = save_dir / f"by_provided_bearid/{threshold_key}"
        save_datasplit(
            df=df_resized,
            save_dir=output_dir,
            chips_root_dir=chips_root_dir,
            threshold_value=threshold_value,
        )


def run(
    chips_root_dir: Path,
    bearid_root_path: Path,
    save_dir: Path,
    thresholds: dict = THRESHOLDS,
) -> None:
    return save_all_datasplits(
        chips_root_dir=chips_root_dir,
        bearid_root_path=bearid_root_path,
        save_dir=save_dir,
        thresholds=thresholds,
    )
