"""Prototype: GroundingDINO proposals + CLIP re-typing for breakfast_box."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DINO_ROOT = "/home/hjh/Desktop/SANAM_compare/UniVAD/models/GroundingDINO"
BERT = "/home/hjh/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
WEIGHTS = ("/home/hjh/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/"
           "snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth")
CONFIG = DINO_ROOT + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"

sys.path.insert(0, DINO_ROOT)


def build_gdino(device):
    from transformers import BertModel
    if not hasattr(BertModel, "get_head_mask"):
        BertModel.get_head_mask = lambda self, head_mask, layers, *unused: [None] * layers
    original = BertModel.get_extended_attention_mask

    def _compat(self, attention_mask, input_shape, device_or_dtype=None):
        dtype = device_or_dtype if isinstance(device_or_dtype, torch.dtype) else self.dtype
        return original(self, attention_mask, input_shape, dtype=dtype)

    BertModel.get_extended_attention_mask = _compat
    from groundingdino.datasets import transforms as T
    from groundingdino.models import build_model
    from groundingdino.models.GroundingDINO import ms_deform_attn as deform
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import get_phrases_from_posmap

    if not hasattr(deform, "_C"):
        class _PytorchDeformableOps:
            @staticmethod
            def ms_deform_attn_forward(value, spatial_shapes, level_start_index,
                                      sampling_locations, attention_weights, im2col_step):
                return deform.multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)

        deform._C = _PytorchDeformableOps()

    config = SLConfig.fromfile(CONFIG)
    config.device = device
    config.text_encoder_type = BERT
    model = build_model(config)
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model = model.to(device).eval()
    return model, T, get_phrases_from_posmap


def detect(model, T, get_phrases, image, caption, device, box_threshold=0.25, text_threshold=0.25):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor, _ = transform(image, None)
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    with torch.inference_mode():
        outputs = model(tensor[None].to(device), captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    keep = logits.max(dim=1)[0] > box_threshold
    logits = logits[keep]
    boxes = boxes[keep]
    tokenized = model.tokenizer(caption)
    phrases = [get_phrases(row > text_threshold, tokenized, model.tokenizer).replace(".", "")
               for row in logits]
    scores = logits.max(dim=1)[0].tolist()
    out = []
    w, h = image.size
    for box, score, phrase in zip(boxes.tolist(), scores, phrases):
        cx, cy, bw, bh = box
        out.append({
            "phrase": phrase,
            "score": score,
            "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h],
        })
    return out


if __name__ == "__main__":
    import open_clip

    device = "cuda:0"
    t0 = time.time()
    model, T, get_phrases = build_gdino(device)
    print("gdino loaded", time.time() - t0, flush=True)
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
    print("clip loaded", time.time() - t0, flush=True)

    VOCAB = {
        "peach": ["peach"],
        "orange": ["orange"],
        "granola": ["granola", "cereal", "muesli"],
        "banana_slice": ["banana slice", "banana"],
        "almond": ["almond", "almonds"],
        "bento_box": ["bento box", "lunch box"],
    }
    caption = ". ".join(
        p for key, ps in VOCAB.items() for p in ps) + "."
    prompts = {k: [f"a photo of {p}" for p in ps] for k, ps in VOCAB.items()}
    flat_prompts = [p for ps in prompts.values() for p in ps]
    keys = [k for k in prompts for _ in prompts[k]]
    with torch.inference_mode():
        text_feat = clip_model.encode_text(tokenizer(flat_prompts).to(device))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    for path in sys.argv[1:]:
        image = Image.open(path).convert("RGB")
        dets = detect(model, T, get_phrases, image, caption, device)
        crops, cboxes = [], []
        for d in dets:
            x0, y0, x1, y1 = d["xyxy"]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.width, x1), min(image.height, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crops.append(clip_pre(image.crop((x0, y0, x1, y1))))
            cboxes.append((x0, y0, x1, y1, d))
        if crops:
            with torch.inference_mode():
                feats = clip_model.encode_image(torch.stack(crops).to(device))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sim = feats @ text_feat.T
                best = sim.argmax(dim=1)
                bestval = sim.max(dim=1).values.tolist()
        print("=" * 90)
        print(path, "raw_dets=", len(dets), "crops=", len(crops))
        rows = []
        for i, (x0, y0, x1, y1, d) in enumerate(cboxes):
            rows.append((keys[best[i]], bestval[i], d["score"], d["phrase"], (x0, y0, x1, y1)))
        for r in sorted(rows, key=lambda r: -r[1]):
            (k, cv, gv, ph, (bx0, by0, bx1, by1)) = r
            print("  %-14s clip=%.3f gd=%.3f phrase=%-22s box=(%4.0f,%4.0f,%4.0f,%4.0f) w=%3.0f h=%3.0f" % (k, cv, gv, ph, bx0, by0, bx1, by1, bx1-bx0, by1-by0))
