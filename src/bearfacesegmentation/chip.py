import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from bearfacesegmentation.predict import predict


def resize(
    mask: np.ndarray,
    dim: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
):
    """Resize the mask to the provided `dim` using the interpolation method.

    `dim`: (W, H) format
    """
    return cv2.resize(mask, dsize=dim, interpolation=interpolation)


def predict_bear_head(
    model: YOLO,
    image_filepath: Path,
    max_det: int = 10,
) -> Optional[Any]:
    try:
        if not image_filepath.exists():
            return None
        else:
            prediction_results = predict(model, image_filepath, max_det=max_det)
            return prediction_results[0]
    except:
        logging.error(
            f"Could not run inference on the following image_filepath: {image_filepath}"
        )
        return None


def crop_from_yolov8(prediction_yolov8) -> np.ndarray:
    """Given a yolov8 prediction, returns an image containing the cropped bear
    head."""
    H, W = prediction_yolov8.orig_shape
    predictions_masks = prediction_yolov8.masks.data.to("cpu").numpy()
    idx = np.argmax(prediction_yolov8.boxes.conf.to("cpu").numpy())
    predictions_mask = predictions_masks[idx]
    prediction_resized = resize(predictions_mask, dim=(W, H))
    masked_image = prediction_yolov8.orig_img.copy()
    black_pixel = [0, 0, 0]
    masked_image[~prediction_resized.astype(bool)] = black_pixel
    x0, y0, x1, y1 = prediction_yolov8.boxes[idx].xyxy[0].to("cpu").numpy()
    return masked_image[int(y0) : int(y1), int(x0) : int(x1)]


def square_pad(img: np.ndarray):
    """Returns an image with dimension max(W, H) x max(W, H), padded with black
    pixels."""
    H, W, _ = img.shape
    K = max(H, W)
    top = (K - H) // 2
    bottom = (K - H) // 2
    left = (K - W) // 2
    right = (K - W) // 2

    return cv2.copyMakeBorder(
        img.copy(),
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
    )
