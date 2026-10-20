"""Open-vocabulary perception and executable symbolic rules for FSAD V12.

GroundingDINO turns an LLM entity description into boxes.  Support images pick
the text prompt and detector threshold, fit slot/shape prototypes and calibrate
normal fact envelopes.  An evolutionary search then builds the normality AST.
No dataset category or anomaly type is hard-coded in this module.
"""

from __future__ import annotations

import math
import re
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

from .causal_evolution import GroupRobustEvolutionSearch
from .features import prepare_inp_aligned_rgb
from .fuzzy_logic import FuzzyLogicProgram
from .orthogonal_symbolic import GenericLogicInference
from .symbolicdet_evolution import EvolutionConfig


DEFAULT_GROUNDING_DINO_ROOT = (
    "/home/hjh/Desktop/SANAM_compare/UniVAD/models/GroundingDINO"
)
DEFAULT_GROUNDING_DINO_CONFIG = (
    DEFAULT_GROUNDING_DINO_ROOT
    + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_GROUNDING_DINO_WEIGHTS = (
    "/home/hjh/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/"
    "snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/"
    "groundingdino_swint_ogc.pth"
)
DEFAULT_BERT = (
    "/home/hjh/.cache/huggingface/hub/models--bert-base-uncased/"
    "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
)


def _install_transformers_compatibility() -> None:
    """Make the original GroundingDINO BERT wrapper work with transformers 5."""
    from transformers import BertModel

    if not hasattr(BertModel, "get_head_mask"):
        BertModel.get_head_mask = (  # type: ignore[attr-defined]
            lambda self, head_mask, layers, *unused: [None] * layers
        )
    if not getattr(BertModel.get_extended_attention_mask, "_ovd_v12_compat", False):
        original = BertModel.get_extended_attention_mask

        def compatible(self, attention_mask, input_shape, device_or_dtype=None):
            dtype = (
                device_or_dtype
                if isinstance(device_or_dtype, torch.dtype)
                else self.dtype
            )
            return original(self, attention_mask, input_shape, dtype=dtype)

        compatible._ovd_v12_compat = True  # type: ignore[attr-defined]
        BertModel.get_extended_attention_mask = compatible


def _install_deformable_attention_fallback(module) -> str:
    """Use torch grid_sample when the locally cached CUDA extension has ABI drift."""
    if hasattr(module, "_C"):
        return "compiled_cuda_extension"

    class PytorchDeformableOps:
        @staticmethod
        def ms_deform_attn_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, im2col_step,
        ):
            return module.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

    module._C = PytorchDeformableOps()
    return "torch_grid_sample_fallback"


def _pil(image: Any) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class GroundingDINOAdapter:
    """Reusable offline GroundingDINO detector with NMS and memoization."""

    def __init__(
        self,
        device: str = "cuda:1",
        root: str = DEFAULT_GROUNDING_DINO_ROOT,
        config: str = DEFAULT_GROUNDING_DINO_CONFIG,
        weights: str = DEFAULT_GROUNDING_DINO_WEIGHTS,
        bert: str = DEFAULT_BERT,
        text_threshold: float = 0.15,
        nms_iou: float = 0.45,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.root = str(Path(root).expanduser().resolve())
        self.config_path = str(Path(config).expanduser().resolve())
        self.weights_path = str(Path(weights).expanduser().resolve())
        self.bert_path = str(Path(bert).expanduser().resolve())
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.cache: Dict[Tuple[str, str], list] = {}

        sys.path.insert(0, self.root)
        _install_transformers_compatibility()
        from groundingdino.datasets import transforms as T
        from groundingdino.models import build_model
        from groundingdino.models.GroundingDINO import ms_deform_attn as deform
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import get_phrases_from_posmap

        self.deformable_backend = _install_deformable_attention_fallback(deform)
        cfg = SLConfig.fromfile(self.config_path)
        cfg.device = self.device
        cfg.text_encoder_type = self.bert_path
        model = build_model(cfg)
        try:
            checkpoint = torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model = model.to(self.device).eval()
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.get_phrases_from_posmap = get_phrases_from_posmap

    def detect(
        self, image: Any, caption: str, *, minimum_score: float = 0.10,
        max_detections: int = 32, cache_id: Optional[str] = None,
        nms_iou: Optional[float] = None,
    ) -> list:
        key_image = cache_id or (
            str(Path(image).resolve())
            if isinstance(image, (str, Path)) else f"memory:{id(image)}"
        )
        caption = caption.lower().strip().rstrip(".") + "."
        key = (key_image, caption)
        if key in self.cache:
            raw = self.cache[key]
        else:
            source = _pil(image)
            tensor, _ = self.transform(source, None)
            with torch.inference_mode():
                outputs = self.model(
                    tensor[None].to(self.device), captions=[caption]
                )
            logits = outputs["pred_logits"].detach().cpu().sigmoid()[0]
            boxes = outputs["pred_boxes"].detach().cpu()[0]
            scores = logits.max(dim=1)[0]
            keep = scores > 0.05
            tokenized = self.model.tokenizer(caption)
            raw = []
            for box, score, token_scores in zip(
                boxes[keep], scores[keep], logits[keep]
            ):
                cx, cy, width, height = [float(v) for v in box]
                phrase = self.get_phrases_from_posmap(
                    token_scores > self.text_threshold,
                    tokenized,
                    self.model.tokenizer,
                ).replace(".", "").strip()
                raw.append({
                    "phrase": phrase,
                    "score": float(score),
                    "box": [
                        cx - width / 2.0, cy - height / 2.0,
                        cx + width / 2.0, cy + height / 2.0,
                    ],
                })
            self.cache[key] = raw
        selected = [row for row in raw if float(row["score"]) >= minimum_score]
        if not selected:
            return []
        boxes = torch.tensor([row["box"] for row in selected], dtype=torch.float32)
        scores = torch.tensor([row["score"] for row in selected], dtype=torch.float32)
        kept = nms(
            boxes, scores, self.nms_iou if nms_iou is None else float(nms_iou)
        ).tolist()[:max_detections]
        return [selected[index] for index in kept]

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def derive_prompt_candidates(entity: Mapping[str, Any]) -> list:
    explicit = entity.get("detector_prompts", [])
    if isinstance(explicit, str):
        explicit = [explicit]
    identifier = str(entity.get("id", "")).replace("_", " ").strip().lower()
    description = str(entity.get("visual_description", "")).strip().lower()
    tokens = identifier.split()
    candidates = [str(item).strip().lower() for item in explicit if str(item).strip()]
    if identifier:
        candidates.append(identifier)
    if tokens:
        noun = tokens[-1]
        candidates.extend([noun, noun[:-1] if noun.endswith("s") else noun + "s"])
        match = re.search(
            rf"(?:\b\w+\s+){{0,2}}\b{re.escape(noun)}\b", description
        )
        if match:
            phrase = re.sub(
                r"^(one|two|three|four|five|six|seven|eight|nine|ten)\s+",
                "", match.group(0),
            )
            candidates.append(phrase)
    result = []
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip(" .")
        if candidate and candidate not in result:
            result.append(candidate)
    return result[:4]


def _ordered(rows: Sequence[Mapping[str, Any]], arrangement: str) -> list:
    if arrangement == "column":
        return sorted(rows, key=lambda row: (row["center"][1], row["center"][0]))
    return sorted(rows, key=lambda row: (row["center"][0], row["center"][1]))


def _descriptor(row: Mapping[str, Any]) -> Dict[str, Any]:
    x0, y0, x1, y1 = [float(v) for v in row["box"]]
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    return {
        **dict(row),
        "center": np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32),
        "size": np.asarray([width, height], dtype=np.float32),
        "aspect": float(math.log(width / height)),
    }


def _pose_normalized(rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return translation/scale-invariant centers and relative box sizes."""
    centers = np.stack([row["center"] for row in rows]).astype(np.float32)
    sizes = np.stack([row["size"] for row in rows]).astype(np.float32)
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    spread = max(float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))), 1e-4)
    canonical_centers = centered / spread
    area_scale = max(float(np.median(np.sqrt(np.prod(sizes, axis=1)))), 1e-4)
    relative_sizes = sizes / area_scale
    return canonical_centers, relative_sizes


def _relational_signature(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Describe an entity set using similarity-transform invariant relations.

    The detector boxes are ordered along their principal axis.  Layout is then
    represented by normalized adjacent gaps and orthogonal deviation, while
    size and aspect are represented relative to the other entities in the same
    image.  No category-specific geometry is encoded here.
    """
    if not rows:
        raise ValueError("a relational signature needs at least one entity")
    centers = np.stack([row["center"] for row in rows]).astype(np.float32)
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    if len(rows) > 1:
        covariance = centered.T @ centered / float(len(rows))
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1].astype(np.float32)
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis
    else:
        axis = np.asarray([1.0, 0.0], dtype=np.float32)
    orthogonal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    projection = centered @ axis
    order = np.argsort(projection, kind="stable")
    ordered_projection = projection[order]
    gaps = np.maximum(np.diff(ordered_projection), 0.0)
    gap_sum = max(float(np.sum(gaps)), 1e-6)
    normalized_gaps = (gaps / gap_sum).astype(np.float32)
    along_scale = max(float(np.std(projection)), 1e-6)
    line_deviation = float(np.std(centered @ orthogonal) / along_scale)

    areas = np.asarray([
        float(np.prod(rows[index]["size"])) for index in order
    ], dtype=np.float32)
    log_areas = np.log(np.maximum(areas, 1e-8))
    relative_log_areas = log_areas - float(np.mean(log_areas))
    aspects = np.asarray([
        float(rows[index]["aspect"]) for index in order
    ], dtype=np.float32)
    relative_aspects = aspects - float(np.mean(aspects))
    return {
        "order": order.astype(np.int64),
        "axis": axis,
        "gaps": normalized_gaps,
        "line_deviation": line_deviation,
        "relative_log_area": relative_log_areas.astype(np.float32),
        "relative_aspect": relative_aspects.astype(np.float32),
    }


def support_augmentations(image: Any) -> list:
    """Deterministic, label-free normal views for detector robustness checks."""
    base = prepare_inp_aligned_rgb(image, input_size=448, crop_size=392)
    height, width = base.shape[:2]
    center = (width / 2.0, height / 2.0)
    views = [np.ascontiguousarray(base[:, ::-1])]
    for factor in (0.78, 1.22):
        views.append(np.clip(base.astype(np.float32) * factor, 0, 255).astype(np.uint8))
    for angle in (-7.0, 7.0):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        views.append(cv2.warpAffine(
            base, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ))
    return views


@dataclass
class OVDGroundingResult:
    facts: Dict[str, float]
    maps: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any]


class OpenVocabularyGrounder:
    FACTS = ("count_error", "layout_error", "size_error", "aspect_error")

    def __init__(self, detector: GroundingDINOAdapter, state: Mapping[str, Any]):
        self.detector = detector
        self.state = dict(state)
        self.prompt = str(state["prompt"])
        self.threshold = float(state["threshold"])
        self.nms_iou = float(state.get("nms_iou", detector.nms_iou))
        self.expected_count = int(state["expected_count"])
        self.arrangement = str(state["arrangement"])
        self.image_size = int(state.get("image_size", 256))
        self.prototype = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in state["prototype"].items()
        }

    @staticmethod
    def _detector_view(image: Any) -> Tuple[np.ndarray, str]:
        if isinstance(image, np.ndarray) and image.shape[:2] == (392, 392):
            aligned = np.asarray(image, dtype=np.uint8)
            digest = hashlib.sha1(aligned.tobytes()).hexdigest()
            return aligned, f"array_sha1:{digest}|inp448_crop392"
        else:
            aligned = prepare_inp_aligned_rgb(image, input_size=448, crop_size=392)
        cache_id = (
            str(Path(image).resolve()) + "|inp448_crop392"
            if isinstance(image, (str, Path)) else f"memory:{id(image)}|inp448_crop392"
        )
        return aligned, cache_id

    @staticmethod
    def fit(
        detector: GroundingDINOAdapter,
        proposal: Mapping[str, Any],
        supports: Sequence[Any],
        image_size: int = 256,
    ) -> Tuple[Optional["OpenVocabularyGrounder"], Dict[str, Any]]:
        entity = proposal.get("primary_entity")
        if not isinstance(entity, Mapping):
            return None, {"abstained": True, "reason": "no_llm_entity"}
        expected = int(entity["expected_count"])
        arrangement = str(entity.get("arrangement", "free"))
        prompts = derive_prompt_candidates(entity)
        thresholds = (0.12, 0.16, 0.20, 0.24, 0.28, 0.34, 0.40)
        nms_values = (0.15, 0.25, 0.35, 0.45, 0.55)
        candidates = []
        all_detections: Dict[str, list] = {}
        for prompt in prompts:
            for nms_iou in nms_values:
                rows = []
                for item in supports:
                    view, cache_id = OpenVocabularyGrounder._detector_view(item)
                    rows.append(detector.detect(
                        view, prompt, minimum_score=0.05, cache_id=cache_id,
                        nms_iou=nms_iou,
                    ))
                all_detections[f"{prompt}|nms={nms_iou}"] = rows
                augmented_rows = []
                for support_index, item in enumerate(supports):
                    for augmentation_index, view in enumerate(support_augmentations(item)):
                        detector_view, augmentation_cache_id = (
                            OpenVocabularyGrounder._detector_view(view)
                        )
                        augmented_rows.append(detector.detect(
                            detector_view, prompt, minimum_score=0.05,
                            cache_id=augmentation_cache_id,
                            nms_iou=nms_iou,
                        ))
                for threshold in thresholds:
                    filtered = [
                        [_descriptor(row) for row in image_rows if row["score"] >= threshold]
                        for image_rows in rows
                    ]
                    counts = np.asarray([len(row) for row in filtered], dtype=float)
                    match = float(np.mean(counts == expected))
                    mae = float(np.mean(np.abs(counts - expected)))
                    selected_scores = [
                        item["score"] for row in filtered for item in row
                    ]
                    confidence = float(np.mean(selected_scores)) if selected_scores else 0.0
                    augmented_counts = np.asarray([
                        sum(float(item["score"]) >= threshold for item in row)
                        for row in augmented_rows
                    ], dtype=float)
                    augmentation_match = float(np.mean(augmented_counts == expected))
                    objective = (
                        25.0 * (1.0 - match) + mae
                        + 4.0 * (1.0 - augmentation_match) - 0.1 * confidence
                    )
                    candidates.append((
                        objective, -match, -augmentation_match, prompt,
                        threshold, nms_iou, filtered,
                    ))
        if not candidates:
            return None, {"abstained": True, "reason": "no_detector_prompts"}
        candidates.sort(key=lambda row: row[:4])
        (
            objective, negative_match, negative_augmentation_match, prompt,
            threshold, nms_iou, rows,
        ) = candidates[0]
        match_rate = float(-negative_match)
        augmentation_match_rate = float(-negative_augmentation_match)
        audit = {
            "abstained": match_rate < 0.75,
            "prompt_candidates": prompts,
            "selected_prompt": prompt,
            "selected_threshold": threshold,
            "selected_nms_iou": nms_iou,
            "support_match_rate": match_rate,
            "support_augmentation_match_rate": augmentation_match_rate,
            "support_counts": [len(row) for row in rows],
            "deformable_backend": detector.deformable_backend,
        }
        if match_rate < 0.75:
            audit["reason"] = "open_vocabulary_count_not_stable_on_3of4_supports"
            return None, audit
        valid = [row for row in rows if len(row) == expected]
        signatures = [_relational_signature(row) for row in valid]
        aligned = [
            [row[int(index)] for index in signature["order"]]
            for row, signature in zip(valid, signatures)
        ]
        prototype = {
            "gaps": np.median(np.stack([
                signature["gaps"] for signature in signatures
            ]), axis=0),
            "line_deviation": np.asarray(float(np.median([
                signature["line_deviation"] for signature in signatures
            ]))),
            "relative_log_area": np.median(np.stack([
                signature["relative_log_area"] for signature in signatures
            ]), axis=0),
            "relative_aspect": np.median(np.stack([
                signature["relative_aspect"] for signature in signatures
            ]), axis=0),
            "absolute_center": np.median(np.stack([
                [item["center"] for item in row] for row in aligned
            ]), axis=0),
            "absolute_size": np.median(np.stack([
                [item["size"] for item in row] for row in aligned
            ]), axis=0),
        }
        state = {
            "kind": "groundingdino_relational_entities_v12",
            "prompt": prompt,
            "threshold": float(threshold),
            "nms_iou": float(nms_iou),
            "expected_count": expected,
            "arrangement": arrangement,
            "image_size": int(image_size),
            "relation_representation": "pca_ordered_similarity_invariant",
            "prototype": {key: value.tolist() for key, value in prototype.items()},
            "support_match_rate": match_rate,
            "support_augmentation_match_rate": augmentation_match_rate,
        }
        return OpenVocabularyGrounder(detector, state), audit

    def state_dict(self) -> Dict[str, Any]:
        return dict(self.state)

    def _box_map(self, box: Sequence[float]) -> np.ndarray:
        size = self.image_size
        x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0.0, 1.0)
        result = np.zeros((size, size), dtype=np.float32)
        left, top = int(x0 * size), int(y0 * size)
        right, bottom = max(left + 1, int(x1 * size)), max(top + 1, int(y1 * size))
        result[top:bottom, left:right] = 1.0
        return result

    def normal_slot_map(self) -> np.ndarray:
        """Union of normal entity slots, used only to calibrate neural gates."""
        result = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for center, slot_size in zip(
            self.prototype["absolute_center"], self.prototype["absolute_size"]
        ):
            pixel_center = tuple(np.rint(center * self.image_size).astype(int))
            width, height = slot_size * self.image_size
            cv2.ellipse(
                result, pixel_center,
                (max(2, int(width / 2)), max(2, int(height / 2))),
                0, 0, 360, 1.0, -1,
            )
        return result

    def evaluate(self, image: Any) -> OVDGroundingResult:
        detector_view, cache_id = self._detector_view(image)
        detected = [
            _descriptor(row) for row in self.detector.detect(
                detector_view, self.prompt, minimum_score=self.threshold,
                cache_id=cache_id, nms_iou=self.nms_iou,
            )
        ]
        count = len(detected)
        size = self.image_size
        maps = {name: np.zeros((size, size), dtype=np.float32) for name in self.FACTS}
        facts = {
            "count_error": float(abs(count - self.expected_count) / max(self.expected_count, 1)),
            "layout_error": 1.0,
            "size_error": 1.0,
            "aspect_error": 1.0,
        }
        unmatched_slots = set(range(self.expected_count))
        unmatched_boxes = set(range(count))
        assignments = []
        signature = None
        if detected:
            source, target = linear_sum_assignment(np.linalg.norm(
                self.prototype["absolute_center"][:, None]
                - np.stack([row["center"] for row in detected])[None, :], axis=2
            ))
            for slot, box_index in zip(source, target):
                slot, box_index = int(slot), int(box_index)
                assignments.append((slot, box_index))
                unmatched_slots.discard(slot)
                unmatched_boxes.discard(box_index)
        for slot in unmatched_slots:
            center = tuple(np.rint(
                self.prototype["absolute_center"][slot] * size
            ).astype(int))
            width, height = self.prototype["absolute_size"][slot] * size
            cv2.ellipse(
                maps["count_error"], center,
                (max(2, int(width / 2)), max(2, int(height / 2))),
                0, 0, 360, 1.0, -1,
            )
        for box_index in unmatched_boxes:
            maps["count_error"] = np.maximum(
                maps["count_error"], self._box_map(detected[box_index]["box"])
            )
        # Geometry is undefined when the detector cannot recover the expected
        # entity set. Count may still be used when it passed augmentation
        # robustness; other atoms explicitly abstain instead of duplicating it.
        if count != self.expected_count:
            facts["layout_error"] = 0.0
            facts["size_error"] = 0.0
            facts["aspect_error"] = 0.0
            return OVDGroundingResult(facts, maps, {
                "prompt": self.prompt,
                "threshold": self.threshold,
                "detected_count": count,
                "detections": detected,
                "assignments": [list(row) for row in assignments],
                "unmatched_slots": sorted(unmatched_slots),
                "unmatched_detections": sorted(unmatched_boxes),
                "relational_abstention": "detected_count_mismatch",
            })
        signature = _relational_signature(detected)
        detected = [detected[int(index)] for index in signature["order"]]
        if signature is not None:
            gap_error = np.abs(
                signature["gaps"] - self.prototype["gaps"]
            )
            line_error = abs(
                float(signature["line_deviation"])
                - float(self.prototype["line_deviation"])
            )
            layout_error = float(np.mean(gap_error) + line_error)
            size_values = np.abs(
                signature["relative_log_area"]
                - self.prototype["relative_log_area"]
            )
            aspect_values = np.abs(
                signature["relative_aspect"]
                - self.prototype["relative_aspect"]
            )
            facts["layout_error"] = layout_error
            facts["size_error"] = float(np.mean(size_values))
            facts["aspect_error"] = float(np.mean(aspect_values))
            for box_index, row in enumerate(detected):
                box_map = self._box_map(row["box"])
                maps["layout_error"] = np.maximum(
                    maps["layout_error"], layout_error * box_map
                )
                maps["size_error"] = np.maximum(
                    maps["size_error"], float(size_values[box_index]) * box_map
                )
                maps["aspect_error"] = np.maximum(
                    maps["aspect_error"], float(aspect_values[box_index]) * box_map
                )
        for name, value in maps.items():
            positive = value[value > 0]
            if positive.size:
                maps[name] = np.clip(
                    cv2.GaussianBlur(value, (0, 0), 2.0)
                    / max(float(np.quantile(positive, 0.99)), 1e-6),
                    0.0, 1.0,
                ).astype(np.float32)
        return OVDGroundingResult(facts, maps, {
            "prompt": self.prompt,
            "threshold": self.threshold,
            "detected_count": count,
            "detections": detected,
            "assignments": [list(row) for row in assignments],
            "unmatched_slots": sorted(unmatched_slots),
            "unmatched_detections": sorted(unmatched_boxes),
        })


OVD_PREDICATES = {
    "count_eq": ("count_error", "cardinality"),
    "arrangement_eq": ("layout_error", "spatial_configuration"),
    "similar_size": ("size_error", "scale_geometry"),
    "similar_orientation": ("aspect_error", "orientation_geometry"),
}


def _violation(value: float, rule: Mapping[str, Any]) -> float:
    excess = max(0.0, float(value) - float(rule["upper"]))
    return float(math.tanh(excess / max(float(rule["soft_scale"]), 1e-8)))


def compile_ovd_program(
    detector: GroundingDINOAdapter,
    proposal: Mapping[str, Any],
    supports: Sequence[Any],
    *,
    image_size: int = 256,
    search_seed: int = 668,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    grounder, grounding_audit = OpenVocabularyGrounder.fit(
        detector, proposal, supports, image_size=image_size
    )
    if grounder is None:
        return {
            "schema_version": "12.0-groundingdino-symbolic",
            "grounder": None, "rules": [], "logic": None, "abstained": True,
            "abstention_reason": grounding_audit.get("reason", "ungrounded_entity"),
        }, {"grounding": grounding_audit, "evolution": None}
    augmented_supports = [
        view for item in supports for view in support_augmentations(item)
    ]
    support_results = [
        grounder.evaluate(item) for item in list(supports) + augmented_supports
    ]
    rules = []
    for predicate, (fact, family) in OVD_PREDICATES.items():
        if fact == "count_error":
            calibration_results = support_results[:len(supports)]
        else:
            calibration_results = [
                row for row in support_results
                if int(row.diagnostics.get("detected_count", -1))
                == grounder.expected_count
            ]
        values = np.asarray([
            row.facts[fact] for row in calibration_results
        ], dtype=float)
        base_values = np.asarray([
            row.facts[fact] for row in support_results[:len(supports)]
        ], dtype=float)
        floor = 0.005 if fact == "count_error" else 0.01
        median = float(np.median(values))
        robust_sigma = 1.4826 * float(np.median(np.abs(values - median)))
        # Every original support must satisfy the rule. Augmentations estimate
        # a robust envelope without allowing one detector failure to make the
        # rule vacuous.
        if fact == "count_error":
            upper = floor
            scale = max(0.5 / max(grounder.expected_count, 1), 0.05)
        else:
            upper = max(
                float(np.max(base_values)) + floor,
                float(np.quantile(values, 0.90)) + 0.5 * max(robust_sigma, floor),
            )
            scale = max(robust_sigma, upper * 0.20, floor)
        rules.append({
            "id": f"ovd_{predicate}", "predicate": predicate,
            "family": family, "fact": fact, "upper": upper,
            "soft_scale": scale, "candidate_origin": "closed_ovd_dsl",
            "calibration_samples": int(len(values)),
            "calibration": "support_max_plus_robust_aug_q90",
        })
    atom_order = [row["id"] for row in rules]
    normal_truths = np.asarray([
        [1.0 - _violation(result.facts[rule["fact"]], rule) for rule in rules]
        for result in support_results
    ], dtype=np.float64)
    rows = list(normal_truths)
    labels = [0] * len(support_results)
    groups = ["normal_support"] * len(support_results)
    for atom_index, rule in enumerate(rules):
        for support_index in range(len(support_results)):
            changed = normal_truths[support_index].copy()
            changed[atom_index] = 0.0
            rows.append(changed); labels.append(1)
            groups.append(f"counterfactual/{rule['predicate']}")
    search = GroupRobustEvolutionSearch(
        atom_order,
        EvolutionConfig(
            population_size=48, generations=35, tournament_size=5,
            elite_count=6, crossover_rate=0.5, mutation_rate=0.3,
            complexity_penalty=0.004, normal_false_positive_penalty=0.2,
            early_stop_patience=10, random_seed=search_seed,
            seed_all_conjunction=False,
        ),
        groups,
        atom_cost=0.015,
        worst_group_weight=0.30,
        normal_invariance_weight=0.25,
    ).run(np.asarray(rows), np.asarray(labels, dtype=np.uint8))
    active = set(search["best_active_atoms"])
    rules = [rule for rule in rules if rule["id"] in active]
    library = {
        "schema_version": "12.0-groundingdino-symbolic",
        "grounder": grounder.state_dict(),
        "rules": rules,
        "logic": {
            "normality_ast": search["best_normality_ast"],
            "anomaly_ast": search["best_anomaly_ast"],
            "semantics": "product_fuzzy_logic",
        },
        "abstained": False,
        "runtime_requires_llm": False,
        "runtime_requires_open_vocabulary_detector": True,
        "category_specific_python_branching": False,
    }
    return library, {
        "grounding": grounding_audit,
        "support_facts": [row.facts for row in support_results],
        "candidate_rules": rules,
        "evolution": search,
    }


class OpenVocabularySymbolicEngine:
    """Duck-compatible with RuleConditionedINPFusion's symbolic engine API."""

    def __init__(self, library: Mapping[str, Any], detector: GroundingDINOAdapter):
        self.library = dict(library)
        self.rules = list(self.library.get("rules", []))
        state = self.library.get("grounder")
        self.grounder = (
            OpenVocabularyGrounder(detector, state)
            if isinstance(state, Mapping) else None
        )
        self.active_ids = [str(row["id"]) for row in self.rules]
        self.program = (
            FuzzyLogicProgram(self.library["logic"]["anomaly_ast"], self.active_ids)
            if self.active_ids else None
        )

    def evaluate(self, image: Any) -> GenericLogicInference:
        size = self.grounder.image_size if self.grounder is not None else 256
        if self.grounder is None or self.program is None:
            zero = np.zeros((size, size), dtype=np.float32)
            return GenericLogicInference(0.0, zero, {}, {}, {}, [], {"abstained": True})
        grounded = self.grounder.evaluate(image)
        scalar_truths, map_truths, atoms, localizers = {}, {}, {}, {}
        calibration_localizers = {}
        for rule in self.rules:
            name, fact = str(rule["id"]), str(rule["fact"])
            violation = _violation(grounded.facts[fact], rule)
            localizer = grounded.maps[fact]
            scalar_truths[name] = 1.0 - violation
            map_truths[name] = 1.0 - violation * localizer
            localizers[name] = localizer
            calibration_localizers[name] = (
                self.grounder.normal_slot_map()
                if fact == "count_error" else localizer
            )
            atoms[name] = {
                "predicate": rule["predicate"], "family": rule["family"],
                "fact": fact, "fact_value": grounded.facts[fact],
                "upper": rule["upper"], "scale": rule["soft_scale"],
                "violation": violation,
            }
        symbolic, trace = self.program.evaluate(scalar_truths)
        symbolic_map, _ = self.program.evaluate(map_truths)
        return GenericLogicInference(
            float(symbolic), np.asarray(symbolic_map, dtype=np.float32),
            grounded.facts, atoms, localizers, trace, grounded.diagnostics,
            calibration_localizers,
        )


__all__ = [
    "GroundingDINOAdapter", "OpenVocabularyGrounder",
    "OpenVocabularySymbolicEngine", "compile_ovd_program",
    "derive_prompt_candidates",
]
