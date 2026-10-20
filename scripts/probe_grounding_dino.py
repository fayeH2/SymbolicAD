#!/usr/bin/env python3
"""Small offline probe for the locally cached GroundingDINO installation."""

from __future__ import annotations

import argparse
import json
import sys

import torch
from PIL import Image


DINO_ROOT = "/home/hjh/Desktop/SANAM_compare/UniVAD/models/GroundingDINO"
CONFIG = DINO_ROOT + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS = (
    "/home/hjh/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/"
    "snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/"
    "groundingdino_swint_ogc.pth"
)
BERT = (
    "/home/hjh/.cache/huggingface/hub/models--bert-base-uncased/"
    "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    args = parser.parse_args()

    sys.path.insert(0, DINO_ROOT)
    # transformers>=5 removed this inference helper used by the original
    # GroundingDINO wrapper. GroundingDINO never supplies a head mask here.
    from transformers import BertModel
    if not hasattr(BertModel, "get_head_mask"):
        BertModel.get_head_mask = lambda self, head_mask, layers, *unused: [None] * layers
    original_extended_mask = BertModel.get_extended_attention_mask
    def _extended_mask_compat(self, attention_mask, input_shape, device_or_dtype=None):
        dtype = device_or_dtype if isinstance(device_or_dtype, torch.dtype) else self.dtype
        return original_extended_mask(self, attention_mask, input_shape, dtype=dtype)
    BertModel.get_extended_attention_mask = _extended_mask_compat
    from groundingdino.datasets import transforms as T
    from groundingdino.models import build_model
    from groundingdino.models.GroundingDINO import ms_deform_attn as deform
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import get_phrases_from_posmap

    if not hasattr(deform, "_C"):
        class _PytorchDeformableOps:
            @staticmethod
            def ms_deform_attn_forward(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, im2col_step,
            ):
                return deform.multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights
                )
        deform._C = _PytorchDeformableOps()

    config = SLConfig.fromfile(CONFIG)
    config.device = args.device
    config.text_encoder_type = BERT
    model = build_model(config)
    checkpoint = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model = model.to(args.device).eval()
    source = Image.open(args.image).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor, _ = transform(source, None)
    caption = args.caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    with torch.inference_mode():
        outputs = model(tensor[None].to(args.device), captions=[caption])
    raw_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    raw_boxes = outputs["pred_boxes"].cpu()[0]
    keep = raw_logits.max(dim=1)[0] > args.box_threshold
    logits = raw_logits[keep].max(dim=1)[0]
    boxes = raw_boxes[keep]
    tokenized = model.tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(row > args.text_threshold, tokenized, model.tokenizer)
        .replace(".", "")
        for row in raw_logits[keep]
    ]
    width, height = source.size
    rows = []
    for box, score, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
        cx, cy, bw, bh = box
        rows.append({
            "phrase": phrase,
            "score": score,
            "box_cxcywh_normalized": [cx, cy, bw, bh],
            "box_xyxy": [
                (cx - bw / 2) * width, (cy - bh / 2) * height,
                (cx + bw / 2) * width, (cy + bh / 2) * height,
            ],
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
