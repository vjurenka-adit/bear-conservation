import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def image_element_to_image_data(e, base_path: Path) -> dict:
    boxes = e.findall("box")
    boxes_data = []

    def to_xy(e) -> dict:
        return {"x": int(e.get("x")), "y": int(e.get("y"))}

    for box in boxes:
        # TODO: improve this parsing section
        htop = [p for p in box.findall("part") if p.get("name") == "htop"][0]
        lear = [p for p in box.findall("part") if p.get("name") == "lear"][0]
        rear = [p for p in box.findall("part") if p.get("name") == "rear"][0]
        nose = [p for p in box.findall("part") if p.get("name") == "nose"][0]
        leye = [p for p in box.findall("part") if p.get("name") == "leye"][0]
        reye = [p for p in box.findall("part") if p.get("name") == "reye"][0]

        data = {
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
        boxes_data.append(data)

    filepath = base_path / e.get("file")

    return {
        "filepath": filepath,
        "bboxes": boxes_data,
    }


def parse_xml(base_path: Path, filepath: Path) -> dict:
    tree = ET.parse(filepath)
    root = tree.getroot()
    images = root.find("images")

    if not images:
        return {"images": []}
    else:
        return {
            "images": [
                image_element_to_image_data(e, base_path=base_path)
                for e in images.findall("image")
            ],
        }


def load_xml(base_path: Path, filepath: Path) -> dict:
    xml_data = parse_xml(base_path=base_path, filepath=filepath)

    # Add image size to image_data
    for image_data in xml_data["images"]:
        image = Image.open(image_data["filepath"])
        width, height = image.size
        image_data["size"] = {"width": width, "height": height}

    return xml_data
