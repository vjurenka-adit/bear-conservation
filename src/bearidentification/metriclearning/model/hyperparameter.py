import random

import numpy as np

SPLIT_TYPE_TO_NUM_CLASSES = {
    "by_individual": 86,
    "by_provided_bearid": 132,
}

data_augmentation_grid = {
    "colorjitter_on": [True, True, True, False],
    "colorjitter_hue_low": list(np.linspace(-0.4, -0.1, 5)),
    "colorjitter_hue_high": list(np.linspace(0.1, 0.4, 5)),
    "colorjitter_saturation_low": list(np.linspace(0.3, 0.9, 10)),
    "colorjitter_saturation_high": list(np.linspace(0.9, 1.5, 10)),
    "rotation_degrees": list(np.linspace(0, 30, 10)),
}

model_grid = {
    "trunk_backbone": [
        "convnext_tiny",
        "convnext_base",
        "convnext_large",
        "resnet18",
        "resnet50",
        # "efficientnet_v2_s",
        # "vit_b_16",
    ],
    "embedder_embedding_size": [2**p for p in range(7, 13)],
    "embedder_hidden_layer_sizes": [[512], [1024], [2048], [512, 1024], [1024, 2048]],
}

loss_grid = {
    "loss_type": ["arcfaceloss"],
    "arcfaceloss_margin": list(np.linspace(10, 100, 30)),
    "arcfaceloss_scale": [2**p for p in range(5, 9)],
}

optimizers_gid = {
    # MetricLoss
    "metric_loss_type": ["adam"],
    "metric_loss_lr": np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=50),
    "metric_loss_weight_decay_on": [True, False],
    "metric_loss_weight_decay": np.logspace(
        np.log10(0.00001),
        np.log10(0.001),
        base=10,
        num=50,
    ),
    # Embedder
    "embedder_type": ["adam"],
    "embedder_lr": np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=50),
    "embedder_weight_decay_on": [True, False],
    "embedder_weight_decay": np.logspace(
        np.log10(0.00001),
        np.log10(0.001),
        base=10,
        num=50,
    ),
    # Trunk
    "trunk_type": ["adam"],
    "trunk_lr": np.logspace(np.log10(0.00001), np.log10(0.001), base=10, num=50),
    "trunk_weight_decay_on": [True, False],
    "trunk_weight_decay": np.logspace(
        np.log10(0.00001),
        np.log10(0.001),
        base=10,
        num=50,
    ),
}


miner_grid = {
    "type": ["batcheasyhardminer"],
    "pos_strategy": ["semihard", "hard"],
    "neg_strategy": ["semihard", "hard"],
}

run_grid = {
    "batch_size": [16, 32, 64],
    "num_epochs": [100, 150, 200, 250, 300],
    "patience": [25, 40, 55],
}

sampler_grid = {
    "type": ["mperclass"],
    "mperclass_m": list(range(2, 9)),
}

# Hyperameter space
grid_space = {
    "run": run_grid,
    "data_augmentation": data_augmentation_grid,
    "model": model_grid,
    "loss": loss_grid,
    "sampler": sampler_grid,
    "optimizers": optimizers_gid,
    "miner": miner_grid,
}


def _draw_params(random_seed: int, grid: dict) -> dict:
    return {k: random.Random(random_seed).choice(list(v)) for k, v in grid.items()}


def make_loss_config(
    loss_grid: dict,
    random_seed: int,
    model_config: dict,
    split_type: str,
) -> dict:
    assert split_type in list(SPLIT_TYPE_TO_NUM_CLASSES.keys())
    params = _draw_params(random_seed=random_seed, grid=loss_grid)
    result = {}
    result["type"] = params["loss_type"]
    if result["type"] == "arcfaceloss":
        result["config"] = {
            "embedding_size": model_config["embedder"]["embedding_size"],
            "num_classes": SPLIT_TYPE_TO_NUM_CLASSES[split_type],
            "margin": float(params["arcfaceloss_margin"]),
            "scale": params["arcfaceloss_scale"],
        }
    return result


def make_model_config(model_grid: dict, random_seed: int) -> dict:
    params = _draw_params(random_seed=random_seed, grid=model_grid)
    result = {}
    default_preprocessing = {
        "values": {"dtype": "float32", "scale": True},
        "crop_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    result["trunk"] = {
        "backbone": params["trunk_backbone"],
        "preprocessing": default_preprocessing,
    }
    result["embedder"] = {
        "embedding_size": params["embedder_embedding_size"],
        "hidden_layer_sizes": params["embedder_hidden_layer_sizes"],
    }
    return result


def make_optimizers_config(
    optimizers_grid: dict,
    random_seed: int,
    loss_config: dict,
) -> dict:
    result = {}
    params = _draw_params(random_seed=random_seed, grid=optimizers_grid)
    result["embedder"] = {
        "type": params["embedder_type"],
        "config": {
            "lr": float(params["embedder_weight_decay"]),
            "weight_decay": float(params["embedder_weight_decay"])
            if params["embedder_weight_decay_on"]
            else 0.0,
        },
    }
    result["trunk"] = {
        "type": params["trunk_type"],
        "config": {
            "lr": float(params["trunk_weight_decay"]),
            "weight_decay": float(params["trunk_weight_decay"])
            if params["trunk_weight_decay_on"]
            else 0.0,
        },
    }
    if loss_config["type"] == "arcfaceloss":
        result["losses"] = {
            "metric_loss": {
                "type": params["metric_loss_type"],
                "config": {
                    "lr": float(params["metric_loss_lr"]),
                    "weight_decay": float(params["metric_loss_weight_decay"])
                    if params["metric_loss_weight_decay_on"]
                    else 0.0,
                },
            }
        }

    return result


def make_miner_config(random_seed: int, miner_grid: dict) -> dict:
    params = _draw_params(random_seed=random_seed, grid=miner_grid)
    result = {
        "type": params["type"],
        "config": {
            "pos_strategy": params["pos_strategy"],
            "neg_strategy": params["neg_strategy"],
        },
    }
    return result


def make_data_augmentation_config(
    data_augmentation_grid: dict,
    random_seed: int,
) -> dict:
    params = _draw_params(random_seed=random_seed, grid=data_augmentation_grid)
    result = {}
    if params["colorjitter_on"]:
        result["colorjitter"] = {
            "hue": [
                float(params["colorjitter_hue_low"]),
                float(params["colorjitter_hue_high"]),
            ],
            "saturation": [
                float(params["colorjitter_saturation_low"]),
                float(params["colorjitter_saturation_high"]),
            ],
        }
    result["rotation"] = {"degrees": float(params["rotation_degrees"])}
    return result


def make_run_config(random_seed: int, run_grid: dict) -> dict:
    params = _draw_params(random_seed=random_seed, grid=run_grid)
    result = {
        "batch_size": int(params["batch_size"]),
        "num_epochs": int(params["num_epochs"]),
        "patience": int(params["patience"]),
    }
    return result


def make_sampler_config(random_seed: int, sampler_grid: dict) -> dict:
    params = _draw_params(random_seed=random_seed, grid=sampler_grid)
    result = {
        "type": params["type"],
        "config": {
            "m": params["mperclass_m"],
        },
    }
    return result


def make_config(grid_space: dict, random_seed: int, split_type: str) -> dict:
    """Main entrypoint to generate a random config for running a train run."""
    model_config = make_model_config(
        model_grid=grid_space["model"],
        random_seed=random_seed,
    )
    loss_config = make_loss_config(
        loss_grid=grid_space["loss"],
        random_seed=random_seed,
        split_type=split_type,
        model_config=model_config,
    )
    optimizers_config = make_optimizers_config(
        optimizers_grid=grid_space["optimizers"],
        random_seed=random_seed,
        loss_config=loss_config,
    )
    data_augmentation_config = make_data_augmentation_config(
        data_augmentation_grid=grid_space["data_augmentation"],
        random_seed=random_seed,
    )
    miner_config = make_miner_config(
        random_seed=random_seed,
        miner_grid=grid_space["miner"],
    )
    sampler_config = make_sampler_config(
        random_seed=random_seed,
        sampler_grid=grid_space["sampler"],
    )
    run_config = make_run_config(
        random_seed=random_seed,
        run_grid=grid_space["run"],
    )
    return {
        **run_config,
        "data_augmentation": data_augmentation_config,
        "model": model_config,
        "loss": loss_config,
        "optimizers": optimizers_config,
        "miner": miner_config,
        "sampler": sampler_config,
    }
