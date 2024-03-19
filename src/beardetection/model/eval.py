import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from tqdm import tqdm
from ultralytics import YOLO

from beardetection.data.utils import yaml_read

LABEL_TO_CLASS = {"other": 1, "bear": 0}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}


def plot_confusion_matrix(
    cf_matrix: np.ndarray,
    normalize=False,
    class_name_mapping: dict = CLASS_TO_LABEL,
    save_path: Optional[Path] = None,
) -> None:
    """Given a confusion matrix `cf_matrix`, it displays it as a pyplot
    graph."""
    cm = cf_matrix.copy()
    ax = None
    if normalize:
        epsilon = 0.0001
        cm = cm.astype(np.float64) / (cm.sum(axis=1)[:, np.newaxis] + epsilon)
        ax = sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues")
    else:
        ax = sns.heatmap(cm, annot=True, cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ## Ticket labels - List must be in alphabetical order
    labels = sorted([class_name_mapping[i] for i in range(2)])
    labels = [class_name_mapping[i] for i in range(2)]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Save the confusion matrix
    if save_path:
        figname = "confusion_matrix_normalized" if normalize else "confusion_matrix"
        fp = save_path / f"{figname}.png"
        logging.info(f"saving confusion matrix in {fp}")
        plt.savefig(str(fp))
    else:
        plt.show()
    plt.close()


def load_model(weights_filepath: Path) -> YOLO:
    assert weights_filepath.exists()
    return YOLO(weights_filepath)


def to_bear_prob(yolov8_predictions) -> float:
    """Returns a float that represents the probability of having a bear from
    the yolov8_predictions.

    Note:
    0.0 means no bear - no bounding boxes were found by yolo.
    """
    if yolov8_predictions is None:
        return 0.0
    elif len(yolov8_predictions) == 0:
        return 0.0
    else:
        yolov8_prediction = yolov8_predictions[0]
        confs = yolov8_prediction.boxes.conf.to("cpu").numpy()
        if len(confs) == 0:
            return 0.0
        else:
            max_conf = np.max(confs)
            assert 0.0 <= max_conf <= 1.0
            return max_conf


def inference_df(model: YOLO, split: str, data_filepath: Path) -> pd.DataFrame:
    data = yaml_read(data_filepath)
    images_dir = data_filepath.parent / data[split]
    labels_dir = images_dir.parent / "labels"
    stem_to_image_filepath = {
        Path(fp).stem: images_dir / fp for fp in os.listdir(images_dir)
    }
    stem_to_label_filepath = {
        Path(fp).stem: labels_dir / fp for fp in os.listdir(labels_dir)
    }
    result = []
    for stem, image_filepath in tqdm(stem_to_image_filepath.items()):
        ground_truth = "bear" if stem in stem_to_label_filepath else "other"
        yolov8_predictions = model.predict(image_filepath)
        prediction = to_bear_prob(yolov8_predictions)
        result.append(
            {
                "image_filepath": image_filepath,
                "ground_truth_label": ground_truth,
                "is_bear_prediction": prediction,
            }
        )
    return pd.DataFrame(result)


def make_y_true(
    df: pd.DataFrame,
    label_to_class: dict = LABEL_TO_CLASS,
) -> torch.Tensor:
    return torch.tensor(
        df["ground_truth_label"].map(lambda label: label_to_class[label]).to_list()
    )


def make_y_hat(
    df: pd.DataFrame,
    bear_threshold: float = 0.0,
    label_to_class: dict = LABEL_TO_CLASS,
) -> torch.Tensor:
    return torch.tensor(
        df["is_bear_prediction"]
        .map(
            lambda x: label_to_class["bear"]
            if x > bear_threshold
            else label_to_class["other"]
        )
        .to_list()
    )


def save_errors(
    model: YOLO,
    df_inference: pd.DataFrame,
    save_path: Path,
    n: int = 10,
    bear_threshold: float = 0.0,
) -> None:
    """Save errors (false positives and negatives)."""
    df_fp = df_inference[
        (df_inference["ground_truth_label"] == "other")
        & (df_inference["is_bear_prediction"] > bear_threshold)
    ]
    df_fn = df_inference[
        (df_inference["ground_truth_label"] == "bear")
        & (df_inference["is_bear_prediction"] <= bear_threshold)
    ]
    image_filepaths_fp = (
        df_fp.sort_values("is_bear_prediction", ascending=False)
        .head(n=n)["image_filepath"]
        .to_list()
    )
    image_filepaths_fn = (
        df_fn.sort_values("is_bear_prediction", ascending=True)
        .head(n=n)["image_filepath"]
        .to_list()
    )
    save_path_fp = save_path / "false_positives"
    save_path_fn = save_path / "false_negatives"

    save_path_fp.mkdir(exist_ok=True, parents=True)
    save_path_fn.mkdir(exist_ok=True, parents=True)

    for image_filepath in image_filepaths_fp:
        model.predict(image_filepath, save=True, project=save_path_fp)

    for image_filepath in image_filepaths_fn:
        model.predict(image_filepath, save=True, project=save_path_fn)


def evaluate(
    df_inference: pd.DataFrame,
    bear_threshold: float = 0.0,
    save_path: Optional[Path] = None,
) -> None:
    y_true = make_y_true(df=df_inference)
    y_hat = make_y_hat(df=df_inference, bear_threshold=bear_threshold)
    cf_matrix = torchmetrics.functional.confusion_matrix(
        preds=y_hat,
        target=y_true,
        task="binary",
    )
    precision = torchmetrics.functional.precision(
        preds=y_hat,
        target=y_true,
        task="binary",
    )
    recall = torchmetrics.functional.recall(
        preds=y_hat,
        target=y_true,
        task="binary",
    )
    f1_score = torchmetrics.functional.f1_score(
        preds=y_hat,
        target=y_true,
        task="binary",
    )
    logging.info(f"bear_threshold: {bear_threshold}")
    results = {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item(),
        "confusion_matrix": cf_matrix.tolist(),
    }
    logging.info(f"results: {results}")

    if save_path:
        df_results = pd.DataFrame([results])
        df_results.to_csv(save_path / "results.csv")

    plot_confusion_matrix(
        cf_matrix=cf_matrix.numpy(), normalize=True, save_path=save_path
    )
    plot_confusion_matrix(
        cf_matrix=cf_matrix.numpy(), normalize=False, save_path=save_path
    )
    plt.show()


def run(
    model: YOLO,
    data_filepath: Path,
    split: str = "test",
    save_path: Optional[Path] = None,
) -> None:
    assert split in ["train", "val", "test"]
    logging.info(f"Running inference on split {split} from {data_filepath}")
    df_inference = inference_df(model=model, split=split, data_filepath=data_filepath)
    bear_threshold = 0.0
    if save_path:
        output_path = save_path / split
        output_path.mkdir(exist_ok=True, parents=True)
        evaluate(
            df_inference=df_inference,
            bear_threshold=bear_threshold,
            save_path=output_path,
        )
        save_errors(
            model=model,
            df_inference=df_inference,
            save_path=output_path,
        )
    else:
        evaluate(df_inference=df_inference, bear_threshold=bear_threshold)


## REPL
# weights_filepath = Path("./data/06_models/beardetection/model/weights/model.pt")
# weights_filepath.exists()

# data_filepath = Path("data/05_model_input/beardetection/upsample/yolov8/data.yaml")
# data_filepath.exists()


# splits = ["train", "val", "test"]
# split = "test"

# data = yaml_read(data_filepath)

# model = load_model(weights_filepath)
# model.info()

# df = inference_df(model=model, split="test", data_filepath=data_filepath)
# df.head()

# bear_threshold = 0.0
# df_fp = df[
#     (df["ground_truth_label"] == "other") & (df["is_bear_prediction"] > bear_threshold)
# ]
# df_fn = df[
#     (df["ground_truth_label"] == "bear") & (df["is_bear_prediction"] <= bear_threshold)
# ]

# len(df_fp)
# len(df_fn)

# df_fp.sort_values("is_bear_prediction", ascending=False).head(n=3)[
#     "image_filepath"
# ].to_list()
# df_fn.sort_values("is_bear_prediction", ascending=True).head(n=3)[
#     "image_filepath"
# ].to_list()


# save_path = Path("./data/08_reporting/beardetection/yolov8/evaluation/")

# save_errors(model=model, df_inference=df, save_path=save_path / "test")

# run(model, data_filepath, split="val", save_path=save_path)
# run(model, data_filepath, split="train", save_path=save_path)
# run(model, data_filepath, split="test", save_path=save_path)
