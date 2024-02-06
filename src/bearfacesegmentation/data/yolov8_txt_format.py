from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# For type hints
Contour = np.ndarray
Mask = np.ndarray
Polygon = np.ndarray


def normalize_polygon(polygon, W: int, H: int):
    """`polygon`: numpy array of shape (_,2) containing the polygon x,y
    coordinates.

    `W`: int - width of the image / mask
    `H`: int - height of the image / mask

    returns a numpy array of shape `polygon.shape` with coordinates that are
    normalized between 0-1.

    Will throw an assertion error if all values of the result do not lie in 0-1.
    """
    copy = np.copy(polygon)
    copy = copy.astype(np.float16)
    copy[:, 0] *= 1 / W
    copy[:, 1] *= 1 / H

    assert (
        (copy >= 0) & (copy <= 1)
    ).all(), f"normalized_polygon values are not all in range 0-1, got: {copy}"

    return copy


def contours_to_polygons(contours: list[Contour]) -> list[Polygon]:
    """Turn a list of contours into a list of polygons."""
    polygons = []
    for contour in contours:
        polygon_list = []
        for point in contour:
            x, y = point[0]
            polygon_list.append(np.array([x, y]))
        polygon = np.array(polygon_list)
        polygons.append(polygon)
    return polygons


def is_contour_area_large_enough(contour: Contour, threshold: int = 200) -> bool:
    return cv2.contourArea(contour) > threshold


def mask_to_contours(mask: Mask) -> list[Contour]:
    """Given a mask, it returns its contours."""
    # Loading the mask in grey, only format supported by cv2.findContours
    mask_grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        mask_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Only keep the areas that are big enough
    valid_contours = [
        contour for contour in contours if is_contour_area_large_enough(contour)
    ]
    return valid_contours


def display_contours(mask: Mask, contours: list[Contour]) -> None:
    """Display the contours information, useful for debugging."""
    N = len(contours)
    plt.subplots(1, N + 1, figsize=(15, 15))

    plt.subplot(1, N + 1, 1)
    plt.imshow(mask)
    plt.title("Mask")

    color = (0, 0, 255)
    thickness = 10
    for i, contour in enumerate(contours):
        image_black = np.zeros(mask.shape, dtype=np.uint8)
        image_contour = cv2.drawContours(image_black, [contour], -1, color, thickness)
        plt.subplot(1, N + 1, i + 2)
        plt.imshow(image_contour)
        plt.title(f"Contour {i}")

    plt.show()


# Yolov8 PyTorch TXT format
def stringify_polygon(polygon: Polygon) -> str:
    """Turns a polygon nd_array into a string in the right YOLOv8 format."""
    return " ".join([f"{x} {y}" for (x, y) in polygon])


def filepath_to_ndarray(filepath: Path) -> np.ndarray:
    return cv2.imread(str(filepath))


def mask_filepath_to_yolov8_format_string(filepath: Path) -> str:
    """Given a `filepath` for an individual mask, it returns a yolov8 format
    string describing the polygons for the segmentation tasks."""
    mask = filepath_to_ndarray(filepath)
    label_class = 0
    H, W, _ = mask.shape
    contours = mask_to_contours(mask)
    polygons = contours_to_polygons(contours)
    normalized_polygons = [normalize_polygon(p, W, H) for p in polygons]
    return "\n".join(
        [f"{label_class} {stringify_polygon(p)}" for p in normalized_polygons]
    )
