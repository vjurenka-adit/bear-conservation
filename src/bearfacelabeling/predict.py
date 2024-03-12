from os import path

import groundingdino.datasets.transforms as T
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    device,
    text_threshold=None,
    with_logits=True,
    cpu_only=False,
    token_spans=None,
):
    assert (
        text_threshold is not None or token_spans is not None
    ), "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = device  # "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append((pred_phrase, logit.max().item()))
            else:
                pred_phrases.append((pred_phrase, -1))
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption), token_span=token_spans
        ).to(
            image.device
        )  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for token_span, logit_phr in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = " ".join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([(phrase, logit.item()) for logit in logit_phr_num])
            else:
                all_phrases.extend([(phrase, -1) for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


def image_to_tensor(image):
    image_pil = image.convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # 3, h, w
    return image_tensor


class Model:
    def __init__(self, model_checkpoint_path, device="cpu"):
        args = SLConfig.fromfile(
            path.join(
                path.dirname(path.abspath(__file__)), "GroundingDINO_SwinT_OGC.py"
            )
        )
        args.device = device  # "cuda", "mps" (for M1/M2) or "cpu"
        self.model = build_model(args)
        self.device = device
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = self.model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        _ = self.model.eval()

    def predict(self, image, text_prompt: str = "bear", token_spans=[[(0, 4)]]):
        box_threshold = 0.27
        text_threshold = 0.25

        image_tensor = image_to_tensor(image)
        boxes_filt, pred_phrases = get_grounding_output(
            self.model,
            image_tensor,
            text_prompt,
            box_threshold,
            self.device,
            text_threshold,
            cpu_only=False,
            token_spans=token_spans,
        )

        if boxes_filt.numel():
            best_guess_index = max(
                range(len(pred_phrases)), key=lambda i: pred_phrases[i][1]
            )
            return boxes_filt[best_guess_index].tolist()

        return None
