import argparse
import logging
from pathlib import Path

from dateutil import parser
from tqdm import tqdm

from beardetection.data.split import split_by_camera_and_date
from beardetection.data.utils import (
    exif,
    get_annotation_filepaths,
    get_image_filepaths_without_bears,
    is_valid_annotation_filepath,
    label_filepath_to_image_filepath,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir-hack-the-planet",
        help="path pointing to the hack the planet dataset",
        default="./data/01_raw/Hack the Planet",
        type=Path,
    )
    parser.add_argument(
        "--input-dir",
        help="path pointing to the yolov8 bbox annotations",
        default="./data/04_feature/beardetection/bearbody/HackThePlanet/",
        type=Path,
    )
    parser.add_argument(
        "--balance",
        help="Should we balance the dataset?",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the split",
        default="./data/04_feature/beardetection/split/",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    return True


def add_datetime(X: list[dict]) -> list[dict]:
    Y = []
    for x in X:
        data = x["exif"]
        year = None
        month = None
        day = None
        hour = None
        minute = None
        if data and data.get("DateTime", None):
            try:
                t = parser.parse(data["DateTime"])
                year = t.year
                month = t.month
                day = t.day
                hour = t.hour
                minute = t.minute
            finally:
                Y.append(
                    {
                        **x,
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "minute": minute,
                    }
                )
    return Y


def add_image_filepath(
    input_dir_hack_the_planet: Path,
    input_dir: Path,
    X: list[dict],
) -> list[dict]:
    return [
        {
            **x,
            "image_filepath": label_filepath_to_image_filepath(
                input_dir_hack_the_planet=input_dir_hack_the_planet,
                input_dir=input_dir,
                label_filepath=x["label_filepath"],
            ),
        }
        for x in X
    ]


def add_camera_id(X: list[dict]) -> list[dict]:
    return [{**x, "camera_id": Path(x["image_filepath"]).parent.name} for x in X]


def make_X(input_dir_hack_the_planet: Path, annotations_input_dir: Path) -> list[dict]:
    """Returns the datapoints, which are a list of dicts containing the
    following keys:

    - class: str - value in {bear, other}
    - label_filepath: Optional[str] - optional filepath that contains the label bbox
    - image_filepath: str
    - camera_id:  str
    - year, month, day, hour, minute: integers
    - exif: dict
    """
    annotation_filepaths = get_annotation_filepaths(input_dir=annotations_input_dir)
    logging.info("loading valid annotation filepaths")
    X_bears = [
        {"label_filepath": label_filepath, "class": "bear"}
        for label_filepath in tqdm(annotation_filepaths)
        if is_valid_annotation_filepath(label_filepath)
    ]
    X_bears = add_image_filepath(
        X=X_bears,
        input_dir_hack_the_planet=input_dir_hack_the_planet,
        input_dir=annotations_input_dir,
    )
    image_filepaths_others = get_image_filepaths_without_bears(
        input_dir_hack_the_planet=input_dir_hack_the_planet
    )

    X_others = [
        {"label_filepath": None, "image_filepath": image_filepath, "class": "other"}
        for image_filepath in image_filepaths_others
    ]

    X_raw = [*X_bears, *X_others]

    X_filtered = [x for x in X_raw if x["image_filepath"]]
    logging.info(f"Found {len(X_raw) - len(X_filtered)} missing images")
    X_exif = [{**x, "exif": exif(x["image_filepath"])} for x in X_filtered]
    X_datetime = add_datetime(X_exif)
    X_camera_id = add_camera_id(X_datetime)
    return X_camera_id


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        input_dir = args["input_dir"]
        output_dir = args["output_dir"]
        input_dir_hack_the_planet = args["input_dir_hack_the_planet"]
        X = make_X(
            input_dir_hack_the_planet=input_dir_hack_the_planet,
            annotations_input_dir=input_dir,
        )
        df_split = split_by_camera_and_date(X=X)
        logging.info(df_split.head(n=10))
        logging.info(df_split.groupby("class").count())
        output_dir.mkdir(exist_ok=True, parents=True)
        df_split.to_csv(output_dir / "data_split.csv", sep=";")


## REPL driven

# data_split_filepath = Path("data/04_feature/beardetection/split2/data_split.csv")
# import pandas as pd

# df_split = pd.read_csv(data_split_filepath, sep=";")

# df_split.groupby("class").count()
# annotations_input_dir = Path("./data/04_feature/beardetection/bearbody/HackThePlanet/")
# input_dir_hack_the_planet = Path("./data/01_raw/Hack the Planet/")
# coco_filepath = input_dir_hack_the_planet / "coco.json"
# images_root_dir = input_dir_hack_the_planet / "images"

# image_filepaths_without_bears = get_image_filepaths_without_bears(input_dir_hack_the_planet=input_dir_hack_the_planet)
# len(image_filepaths_without_bears)
# image_filepaths_without_bears[0]

# X = make_X(
#     input_dir_hack_the_planet=input_dir_hack_the_planet,
#     annotations_input_dir=annotations_input_dir,
# )

# annotation_filepaths = get_annotation_filepaths(input_dir=annotations_input_dir)
# len(annotation_filepaths)

# X_bears = [{"label_filepath": label_filepath, "class": "bear"} for label_filepath in annotation_filepaths]
# X_bears = add_image_filepath(
#     X=X_bears,
#     input_dir_hack_the_planet=input_dir_hack_the_planet,
#     input_dir=annotations_input_dir,
# )

# X_others = [{"label_filepath": None, "image_filepath": image_filepath, "class": "other"} for image_filepath in image_filepaths_without_bears]
# X_others[:2]

# X = [*X_bears, *X_others]
# len(X), len(X_bears), len(X_others)

# X_filtered = [x for x in X if x["image_filepath"]]
# logging.info(f"Found {len(X) - len(X_filtered)} missing images")
# X_exif = [{**x, "exif": exif(x["image_filepath"])} for x in X_filtered]
# X_datetime = add_datetime(X_exif)
# X_camera_id = add_camera_id(X_datetime)

# X_camera_id[:5]

# import pandas as pd

# X[:1]
# n = len(X)
# train_ratio = 0.8
# val_ratio = 0.5
# train_size = int(train_ratio * n)
# val_size = int(val_ratio * (n - train_size))
# df = pd.DataFrame(X)
# df.head()

# g = df.groupby(["year", "month", "day", "camera_id"])
# g.head()
# # key -> list of indices
# g.groups

# G = list(g.groups.items())
# random_seed = 0
# import random

# random.Random(random_seed).shuffle(G)
# list(g.groups.items())[3]
# idx = list(g.groups.items())[3][1]
# idx2 = list(g.groups.items())[4][1]
# idx
# idx2
# idx_merge = idx.copy().append(idx2)
# idx_merge
# idx.append(idx2)
# df.iloc[idx]

# len(list(g.groups.items())[0][1])

# n = len(G)
# train_ratio = 0.8
# val_ratio = 0.5
# train_size = int(train_ratio * n)
# val_size = int(val_ratio * (n - train_size))

# G_train = G[:train_size]
# G_val = G[train_size : train_size + val_size]
# G_test = G[train_size + val_size :]

# G_train
# G_val
# G_test

# type(G_train)


# indices = []
# # result = pd.Index([], dtype="int64")
# for _, idx in G_test:
#     print(idx)
#     # result.append(idx)
#     indices.extend(list(idx))
#     # result.extend(idx)
# result = pd.Index(indices, dtype="int64")


# len(to_indices(G_val))
# len(to_indices(G_train))
# len(to_indices(G_test))

# df.iloc[result]

# df_train = df.iloc[to_indices(G_train)].copy()
# df_val = df.iloc[to_indices(G_val)].copy()
# df_test = df.iloc[to_indices(G_test)].copy()
# df_train.head()
# df_train["split"] = "train"
# df_val["split"] = "val"
# df_test["split"] = "test"

# df_split = pd.concat([df_train, df_val, df_test])
# df_split.info()
# df_train


# df[]

# list(G_test[0][1])

# result
