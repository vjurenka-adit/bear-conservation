from pathlib import Path

from ultralytics import YOLO


def load_model(model_str: str) -> YOLO:
    """Loads the `model`"""
    return YOLO(model_str)


def train(model: YOLO, params: dict):
    """Main function for running a train run."""
    data_path = Path(params["data"]).absolute()
    project = f"data/06_models/bearfacedetection/yolov8/"
    experiment_name = params["experiment_name"] or "train"
    model.train(
        project=project,
        name=experiment_name,
        data=data_path,
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        close_mosaic=params["close_mosaic"],
        # Data Augmentation parameters
        degrees=params["degrees"],
        flipud=params["flipud"],
        translate=params["translate"],
    )
