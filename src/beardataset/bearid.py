import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def parse_bbox(box) -> dict:
    """Given a bbox element, returns a dict."""

    def to_xy(e) -> dict:
        return {"x": int(e.get("x")), "y": int(e.get("y"))}

    # TODO: improve this
    htop = [p for p in box.findall("part") if p.get("name") in {"htop", "head_top"}][0]

    lear = box.find(".//part[@name='lear']")
    rear = box.find(".//part[@name='rear']")
    leye = box.find(".//part[@name='leye']")
    reye = box.find(".//part[@name='reye']")
    nose = box.find(".//part[@name='nose']")

    return {
        "label": box.find("label").text,
        "top": int(box.get("top")),
        "left": int(box.get("left")),
        "width": int(box.get("width")),
        "height": int(box.get("height")),
        "parts": {
            "htop": to_xy(htop),
            "lear": to_xy(lear),
            "rear": to_xy(rear),
            "nose": to_xy(nose),
            "leye": to_xy(leye),
            "reye": to_xy(reye),
        },
    }


def parse_image(e, base_path: Path) -> dict:
    """Turns an xml element `e` into a python dict containing the following
    keys:

    - `filepath`: image filepath
    - `bboxes`: list of dicts containing the following keys: top, left, width, height, parts.
    """
    return {
        "filepath": base_path / e.get("file"),
        "bboxes": [parse_bbox(box) for box in e.findall("box")],
    }


def parse_chip(e, base_path: Path) -> dict:
    """Turns an xml element `e` into a python dict containing the following
    keys:

    - `filepath`: Path - chip filepath
    - `label`: str - id of the bear
    - `resolution`: int - image resolution
    - `dimensions`: dict - chip dimension
    - `box`: dict - result of parse_bbox
    - `source`: Path - image filepath
    """

    width_str, height_str = e.find("chip_dimensions").text.split(" ")
    source = e.find("source")

    return {
        "label": e.find("label").text,
        "resolution": int(e.find("resolution").text),
        "filepath": base_path / e.get("file"),
        "source": base_path / source.get("file"),
        "box": parse_bbox(source.find("box")),
        "dimensions": {"width": int(width_str), "height": int(height_str)},
    }


def parse_images_xml(base_path: Path, filepath: Path) -> dict:
    """Parses the filepath and returns a python dict."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    images = root.find("images")

    if not images:
        return {"images": []}
    else:
        return {
            "images": [
                parse_image(e, base_path=base_path) for e in images.findall("image")
            ],
        }


def parse_chips_xml(base_path: Path, filepath: Path) -> dict:
    """Parses the filepath and returns a python dict."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    chips = root.find("chips")

    if not chips:
        return {"chips": []}
    else:
        return {
            "chips": [
                parse_chip(e, base_path=base_path) for e in chips.findall("chip")
            ],
        }


def load_images_xml(base_path: Path, filepath: Path) -> dict:
    """Loads an xml images file from the BearID folder and returns the parsed
    xml data as a python dict.

    Eg.
    >>> load_xml(Path('./data/01_raw/BearID'), Path('./data/01_raw/BearID/images_train_without_bc.xml'))
    """
    xml_data = parse_images_xml(base_path=base_path, filepath=filepath)

    # Add image size to image_data
    for image_data in xml_data["images"]:
        image = Image.open(image_data["filepath"])
        width, height = image.size
        image_data["size"] = {"width": width, "height": height}

    return xml_data
