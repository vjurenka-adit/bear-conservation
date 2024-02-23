import hashlib
import logging
import os
from pathlib import Path

import torch
from lightglue import ALIKED, DISK, SIFT, SuperPoint
from lightglue.utils import load_image
from tqdm import tqdm


def get_filepaths(root: Path, allowed_suffixes={".jpg", ".JPG"}) -> list[Path]:
    """Lists all filepaths given a root directory `root`."""
    return [p for p in root.rglob("*") if p.suffix in allowed_suffixes and p.exists()]


def extractor_type_to_extractor(device, extractor_type: str, n_keypoints: int = 1024):
    """Given an extractor_type in {'sift', 'superpoint', 'aliked', 'disk'},
    returns an extractor."""
    if extractor_type == "sift":
        return SIFT(max_num_keypoints=n_keypoints).eval().to(device)
    elif extractor_type == "superpoint":
        return SuperPoint(max_num_keypoints=n_keypoints).eval().to(device)
    elif extractor_type == "disk":
        return DISK(max_num_keypoints=n_keypoints).eval().to(device)
    elif extractor_type == "aliked":
        return ALIKED(max_num_keypoints=n_keypoints).eval().to(device)
    else:
        raise Exception("extractor_type is not valid")


def extract_all(
    device,
    image_filepaths: list[Path],
    extractor,
) -> dict:
    """Given a list of image filepaths and an extractor, it returns a
    dictionnary of keypoints keyed by image filepath."""
    results = {}

    for image_filepath in tqdm(image_filepaths):
        logging.info(f"Extracting features from {image_filepath}")
        image = load_image(image_filepath).to(device)
        features = extractor.extract(image)
        results[str(image_filepath)] = features

    return results


def md5_filepaths(filepaths: list[Path]) -> str:
    """Given a list of filepaths, it returns the md5 of the concatenated string
    of filepaths."""
    s = "".join([str(fp) for fp in filepaths])
    return str(hashlib.md5(s.encode("utf-8")).hexdigest())


def save(
    filepaths: list[Path],
    save_dir: Path,
    features_dict: dict,
    extractor_type: str,
    n_keypoints: int,
) -> None:
    md5_str = md5_filepaths(filepaths)
    output_filepath = (
        save_dir / f"{md5_str}/{extractor_type}/{n_keypoints}/features.pth"
    )
    os.makedirs(output_filepath.parent, exist_ok=True)
    logging.info(f"Saving features_dict in {output_filepath}")
    torch.save(obj=features_dict, f=output_filepath)


def run(
    chips_root_dir: Path,
    save_dir: Path,
    extractor_type: str,
    n_keypoints: int,
    device,
) -> None:
    filepaths = get_filepaths(root=chips_root_dir)
    extractor = extractor_type_to_extractor(
        device=device, extractor_type=extractor_type, n_keypoints=n_keypoints
    )
    features_dict = extract_all(
        device=device, image_filepaths=filepaths, extractor=extractor
    )
    save(
        filepaths=filepaths,
        save_dir=save_dir,
        features_dict=features_dict,
        extractor_type=extractor_type,
        n_keypoints=n_keypoints,
    )


## REPL
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device

# chips_root_dir = Path(
#     "data/07_model_output/bearfacesegmentation/chips/yolov8/resized/square_dim_300/"
# )
# save_dir = Path("data/04_feature/bearidentification/lightglue/")

# run(
#     chips_root_dir=chips_root_dir,
#     save_dir=save_dir,
#     extractor_type="superpoint",
#     n_keypoints=1024,
#     device=device,
# )
