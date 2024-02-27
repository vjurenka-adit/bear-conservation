import logging
from pathlib import Path

import torch
from lightglue import LightGlue


def load_features_dict(
    root_dir: Path,
    extractor_type: str = "sift",
    n_keypoints: int = 1024,
) -> dict:
    """Loads the cached features_dict for all the generated chips.

    Returns an empty dict if it cannot load them.
    """
    features_filepath = root_dir / f"{extractor_type}/{n_keypoints}/features.pth"
    if not features_filepath.exists():
        logging.error(f"Can not load features_dict from {features_filepath}")
        return {}
    else:
        return torch.load(features_filepath)


def load_matcher(device, features_type: str = "sift"):
    """Loads the matcher (LightGlue) using the features_type in {sift, disk,
    aliked, superpoint}."""
    matcher = LightGlue(features=features_type).eval()
    if device == torch.device("cpu"):
        logging.info("Loading matcher on the cpu")
        return matcher
    else:
        logging.info("Loading matcher on the gpu")
        return matcher.cuda()


def load(
    features_root_dir: Path,
    device,
    n_keypoints: int = 1024,
    extractor_type: str = "sift",
) -> dict:
    """
    Returns a dictionary with the following keys:
    - matcher: LightGlue matcher
    - features_dict: precomputed features keyed by image filepath
    """
    matcher = load_matcher(device=device, features_type=extractor_type)
    features_dict = load_features_dict(
        root_dir=features_root_dir,
        extractor_type=extractor_type,
        n_keypoints=n_keypoints,
    )
    return {
        "matcher": matcher,
        "features_dict": features_dict,
    }


def match_pair(
    path0: Path,
    path1: Path,
    matcher,
    features_dict: dict,
) -> dict:
    """Runs the matcher on path0 and path1 using the features_dict to retrieve
    the features.

    Returns the ouptut of the matcher as a dict.
    """
    feats0 = features_dict[str(path0)]
    feats1 = features_dict[str(path1)]
    return matcher({"image0": feats0, "image1": feats1})


## -----------
## REPL Driven
## -----------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device

# matcher = load_matcher(device)
# matcher

# extractor_type = "sift"
# n_keypoints = 1024
# root_dir = Path(
#     f"data/04_feature/bearidentification/lightglue/8a505e53d07ef04bda85260ece34238d"
# )
# features_dict = load_features_dict(
#     root_dir=root_dir, extractor_type=extractor_type, n_keypoints=n_keypoints
# )
# loaded = load(
#     features_root_dir=root_dir,
#     device=device,
#     n_keypoints=n_keypoints,
#     extractor_type=extractor_type,
# )

# loaded["matcher"]

# chips_root_dir = Path(
#     "data/07_model_output/bearfacesegmentation/chips/yolov8/resized/square_dim_300/"
# )
# chips_root_dir.exists()

# s = random.sample(features_dict.keys(), 1)
# s

# path0 = chips_root_dir / "brooksFalls/bear_mon_201409/bf_089/P1190027.jpg"
# path0.exists()

# path1 = chips_root_dir / "brooksFalls/bear_mon_201409/bf_089/P1190029.jpg"
# path1.exists()

# features_dict[str(path1)]
# ss = set(features_dict.keys())
# len(ss)
# ss

# out = match_pair(
#     path0=path0,
#     path1=path1,
#     matcher=loaded["matcher"],
#     features_dict=loaded["features_dict"],
# )

# out

# from sys import getsizeof

# getsizeof(out)
