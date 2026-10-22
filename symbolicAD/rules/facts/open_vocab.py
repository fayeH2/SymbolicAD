"""GroundingDINO 开放词表实体提案。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

_DINO_ROOT = Path("/home/hjh/Desktop/SANAM_compare/UniVAD/models/GroundingDINO")
_DINO_CONFIG = _DINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
_DINO_WEIGHTS = Path(
    "/home/hjh/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/"
    "snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth"
)
_BERT_COMPAT_PATCHED = False

_DINO_BERT = (
    "/home/hjh/.cache/huggingface/hub/models--bert-base-uncased/"
    "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
)


@dataclass(frozen=True)
class Vocabulary:
    entity_types: Mapping[str, Tuple[str, ...]]

    def flat_phrases(self) -> Tuple[str, ...]:
        rows = []
        for phrases in self.entity_types.values():
            for phrase in phrases:
                if phrase not in rows:
                    rows.append(phrase)
        return tuple(rows)

    def caption(self) -> str:
        return ". ".join(self.flat_phrases()) + "."

    def match(self, phrase: str) -> Optional[str]:
        lowered = str(phrase).strip().lower()
        best = None
        for entity_type, phrases in self.entity_types.items():
            for phrase_index, needle in enumerate(phrases):
                position = lowered.find(needle.lower())
                if position < 0:
                    continue
                key = (position, phrase_index)
                if best is None or key < best[0]:
                    best = (key, entity_type)
        return None if best is None else best[1]


@dataclass(frozen=True)
class Detection:
    entity_type: str
    phrase: str
    score: float
    box_xyxy: Tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box_xyxy
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass(frozen=True)
class DetectionBatch:
    detections: Tuple[Detection, ...]
    rejected: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


def _box_iou(left, right) -> float:
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _nms_per_type(detections, iou_threshold: float):
    by_type = {}
    for detection in detections:
        by_type.setdefault(detection.entity_type, []).append(detection)
    kept = []
    for rows in by_type.values():
        ordered = sorted(rows, key=lambda row: row.score, reverse=True)
        accepted = []
        for row in ordered:
            if all(_box_iou(row.box_xyxy, other.box_xyxy) <= iou_threshold for other in accepted):
                accepted.append(row)
        kept.extend(accepted)
    return tuple(sorted(kept, key=lambda row: (row.entity_type, row.box_xyxy[1], row.box_xyxy[0])))


class GroundingDinoDetector:
    def __init__(
        self,
        device: str = "cuda:0",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_iou: float = 0.55,
        min_image_area_fraction: float = 1e-4,
        max_image_area_fraction: float = 0.6,
        area_fraction_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
        image_short_side: int = 800,
        image_max_side: int = 1333,
    ):
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.min_image_area_fraction = float(min_image_area_fraction)
        self.max_image_area_fraction = float(max_image_area_fraction)
        self.image_short_side = int(image_short_side)
        self.image_max_side = int(image_max_side)
        # Large containers and small parts cannot share one meaningful area
        # range.  Overrides are explicit per typed entity and are serialized by
        # the experiment runner; there is no category-name special case here.
        self.area_fraction_bounds = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in (area_fraction_bounds or {}).items()
        }
        self._model = None
        self._transform = None
        self._get_phrases = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        if str(_DINO_ROOT) not in sys.path:
            sys.path.insert(0, str(_DINO_ROOT))
        from transformers import BertModel

        global _BERT_COMPAT_PATCHED
        if not _BERT_COMPAT_PATCHED:
            # transformers>=5 removed this inference helper that the original
            # GroundingDINO wrapper calls. Patch exactly once: re-wrapping an
            # already-patched method would recurse into itself.
            if not hasattr(BertModel, "get_head_mask"):
                BertModel.get_head_mask = (
                    lambda self, head_mask, layers, *unused: [None] * layers
                )
            original_mask = BertModel.get_extended_attention_mask

            def _extended_mask(self, attention_mask, input_shape, device_or_dtype=None):
                dtype = (
                    device_or_dtype
                    if isinstance(device_or_dtype, torch.dtype)
                    else self.dtype
                )
                return original_mask(self, attention_mask, input_shape, dtype=dtype)

            BertModel.get_extended_attention_mask = _extended_mask
            _BERT_COMPAT_PATCHED = True

        from groundingdino.datasets import transforms as transforms
        from groundingdino.models import build_model
        from groundingdino.models.GroundingDINO import ms_deform_attn as deform
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import get_phrases_from_posmap

        if not hasattr(deform, "_C"):
            class _PytorchDeformableOps:
                @staticmethod
                def ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
                    return deform.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
            deform._C = _PytorchDeformableOps()

        config = SLConfig.fromfile(str(_DINO_CONFIG))
        config.device = self.device
        config.text_encoder_type = _DINO_BERT
        model = build_model(config)
        checkpoint = torch.load(str(_DINO_WEIGHTS), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self._model = model.to(self.device).eval()
        self._transform = transforms.Compose([
            transforms.RandomResize([self.image_short_side], max_size=self.image_max_side),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._get_phrases = get_phrases_from_posmap
        self._tokenizer = self._model.tokenizer

    def detect(self, image: Image.Image, vocabulary: Vocabulary) -> DetectionBatch:
        self._load()
        tensor, _ = self._transform(image, None)
        caption = vocabulary.caption().lower()
        with torch.inference_mode():
            outputs = self._model(tensor[None].to(self.device), captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > self.box_threshold
        logits = logits[keep]
        boxes = boxes[keep]
        tokenized = self._tokenizer(caption)
        width, height = image.size
        image_area = float(width * height)

        detections = []
        rejected = []
        for row_logits, box in zip(logits, boxes):
            positions = row_logits > self.text_threshold
            phrase = self._get_phrases(positions, tokenized, self._tokenizer).replace(".", "").strip()
            score = float(row_logits.max())
            cx, cy, bw, bh = box.tolist()
            xyxy = (
                max(0.0, (cx - bw / 2) * width), max(0.0, (cy - bh / 2) * height),
                min(float(width), (cx + bw / 2) * width),
                min(float(height), (cy + bh / 2) * height),
            )
            entity_type = vocabulary.match(phrase)
            area_fraction = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]) / image_area
            if entity_type is None:
                rejected.append({"reason": "no_vocabulary_match", "phrase": phrase})
                continue
            lower_area, upper_area = self.area_fraction_bounds.get(
                entity_type,
                (self.min_image_area_fraction, self.max_image_area_fraction),
            )
            if not (lower_area <= area_fraction <= upper_area):
                rejected.append({"reason": "area_out_of_range", "phrase": phrase, "entity_type": entity_type, "area_fraction": area_fraction})
                continue
            detections.append(Detection(entity_type, phrase, score, xyxy))
        return DetectionBatch(_nms_per_type(detections, self.nms_iou), tuple(rejected))


__all__ = ["Vocabulary", "Detection", "DetectionBatch", "GroundingDinoDetector"]
