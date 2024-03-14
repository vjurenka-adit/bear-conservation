import logging
from os import path
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import bearfacelabeling.predict


def bbox_to_string(bbox: list[float]) -> str:
    x, y, width, height = bbox
    return f"{x} {y} {width} {height}"


def parse_bbox(s: str) -> list[float]:
    x, y, width, height = [float(x) for x in s.split(" ")]
    return [x, y, width, height]


def load_groundingDINO_model(device: str, model_checkpoint_path: Path):
    return bearfacelabeling.predict.Model(
        model_checkpoint_path,
        device=device,
    )


def annotate(
    model,
    text_prompt: str,
    token_spans: list,
    image_paths: list[str],
    input_dir: Path,
    output_dir: Path,
) -> None:
    for image_path in tqdm(image_paths):
        try:
            image = Image.open(image_path)
            bbox = model.predict(
                image,
                text_prompt=text_prompt,
                token_spans=token_spans,
            )

            relative_image_path = path.relpath(image_path, input_dir)
            relative_ann_path = path.splitext(relative_image_path)[0] + ".txt"
            ann_path = path.join(output_dir, relative_ann_path)

            # Create dir
            Path(path.dirname(ann_path)).mkdir(parents=True, exist_ok=True)

            with open(ann_path, "w") as f:
                if bbox:
                    f.write(bbox_to_string(bbox))
        except:
            logging.warning(f"image {image_path} cannot be read or processed")


def parse_annotations(
    input_dir: Path,
    output_dir: Path,
    image_paths: list[Path],
) -> dict[str, pd.DataFrame]:
    anns = {}
    imgs_without_ann_file = []

    for img_path in image_paths:
        img_rel_path = relative_image_path = path.relpath(img_path, input_dir)
        ann_rel_path = path.splitext(img_rel_path)[0] + ".txt"

        try:
            with open(path.join(output_dir, ann_rel_path), "r") as f_ann:
                content = f_ann.read()
                if not content:
                    anns[img_rel_path] = None
                else:
                    lines = content.split("\n")
                    bboxes = [parse_bbox(line) for line in lines]
                    if bboxes:
                        anns[img_rel_path] = bboxes[0]
                    else:
                        anns[img_rel_path] = None
        except FileNotFoundError as err:
            imgs_without_ann_file.append(img_rel_path)

    df_image_pb = pd.DataFrame.from_dict({"img": imgs_without_ann_file})
    df = pd.DataFrame.from_dict({"img": anns.keys(), "bbox": anns.values()})

    return {"ok": df, "ko": df_image_pb}
