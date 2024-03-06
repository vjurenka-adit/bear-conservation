import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from bearidentification.metriclearning.eval import run
from bearidentification.metriclearning.utils import yaml_read


def make_df_experiment(evaluation_root_dir: Path) -> pd.DataFrame:
    print(evaluation_root_dir)
    evaluation_metrics_filepath = evaluation_root_dir / "evaluation" / "metrics.csv"
    print(evaluation_metrics_filepath)
    splits = ["train", "val", "test"]
    metrics_columns = [
        "precision_at_1_level0",
        "precision_at_3_level0",
        "precision_at_5_level0",
        "precision_at_10_level0",
    ]
    df_evaluation_metrics = pd.read_csv(evaluation_metrics_filepath)
    args = yaml_read(evaluation_root_dir / "args.yaml")
    data = {
        "experiment_name": args["run"]["experiment_name"],
        "datasplit_type": args["run"]["datasplit"]["split_type"],
        "datasplit_size": args["run"]["datasplit"]["dataset_size"],
        "datasplit_root_dir": args["run"]["datasplit"]["split_root_dir"],
        "batch_size": args["batch_size"],
        "num_epochs": args["num_epochs"],
        "patience": args["patience"],
        "data_augmentation": args.get("data_augmentation", {}),
        "trunk": args["model"]["trunk"]["backbone"],
        "embedder": args["model"]["embedder"]["embedding_size"],
        "loss": args["loss"]["type"],
        "sampler": args["sampler"]["type"] if args.get("sampler", None) else None,
        "optimizer_trunk": args["optimizers"]["trunk"]["type"],
        "optimizer_trunk_lr": args["optimizers"]["trunk"]["config"]["lr"],
        "optimizer_trunk_weight_decay": args["optimizers"]["trunk"]["config"][
            "weight_decay"
        ],
        "optimizer_embedder": args["optimizers"]["embedder"]["type"],
        "optimizer_embedder_lr": args["optimizers"]["embedder"]["config"]["lr"],
        "optimizer_embedder_weight_decay": args["optimizers"]["embedder"]["config"][
            "weight_decay"
        ],
        "miner": args["miner"]["type"] if args.get("miner", None) else None,
    }

    available_splits = set(df_evaluation_metrics["split"].unique())
    for split in splits:
        for metrics_column in metrics_columns:
            if split in available_splits:
                data[f"{split}_{metrics_column}"] = df_evaluation_metrics[
                    df_evaluation_metrics["split"] == split
                ].iloc[0][metrics_column]
            else:
                data[f"{split}_{metrics_column}"] = None

    return pd.DataFrame.from_dict([data])


def make_df_summary(evaluations_root_dir: Path) -> pd.DataFrame:
    paths = [
        evaluations_root_dir / subdir
        for subdir in os.listdir(evaluations_root_dir)
        if os.path.isdir(evaluations_root_dir / subdir)
    ]
    df_summary = pd.concat([make_df_experiment(evaluation_root_dir=p) for p in paths])
    df_summary.reset_index(drop=True)
    return df_summary


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser.

    Hyperparameters can be passed for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluations-root-dir",
        help="root path to save the eval run metrics.",
        type=Path,
        required=True,
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
    # TODO
    return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        evaluations_root_dir = args["evaluations_root_dir"]
        logging.info(f"Generating a summary of all models")
        print(evaluations_root_dir)
        df_summary = make_df_summary(evaluations_root_dir=evaluations_root_dir)
        df_summary.to_csv(evaluations_root_dir / "summary.csv")
