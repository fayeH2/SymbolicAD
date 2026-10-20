"""Offline normality-rule proposal with the local Qwen3-VL model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .features import FEATURE_CATALOG, compact_feature_catalog, prepare_inp_aligned_rgb


DEFAULT_MODEL_PATH = "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"


def _heatmap_panel(
    score_map: np.ndarray, size: Tuple[int, int], value_range: Tuple[float, float] = None,
) -> Image.Image:
    score = np.asarray(score_map, dtype=np.float32).squeeze()
    minimum, maximum = value_range or (float(score.min()), float(score.max()))
    normalized = np.zeros_like(score) if maximum <= minimum else (score - minimum) / (maximum - minimum)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored).resize(size, Image.Resampling.BILINEAR)


def make_support_montage(
    image_paths: Sequence[Union[str, Path]], anomaly_maps: Sequence[np.ndarray],
    output_path: Union[str, Path], tile_size: int = 256,
) -> Path:
    if len(image_paths) != len(anomaly_maps):
        raise ValueError("support image/map count mismatch")
    flattened = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1)
                                for item in anomaly_maps])
    shared_range = tuple(float(value) for value in np.quantile(flattened, [0.01, 0.99]))
    rows = max(1, len(image_paths))
    canvas = Image.new("RGB", (tile_size * 2, rows * (tile_size + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, score_map) in enumerate(zip(image_paths, anomaly_maps)):
        original = Image.fromarray(prepare_inp_aligned_rgb(path))
        original.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (tile_size, tile_size), "black")
        x_offset = (tile_size - original.width) // 2
        y_offset = (tile_size - original.height) // 2
        panel.paste(original, (x_offset, y_offset))
        heatmap = _heatmap_panel(score_map, (tile_size, tile_size), shared_range)
        top = index * (tile_size + 28)
        canvas.paste(panel, (0, top))
        canvas.paste(heatmap, (tile_size, top))
        draw.text((5, top + tile_size + 5), f"normal support {index + 1}: image | INP map", fill="black")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _support_statistics(rows: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    result = {}
    for feature in FEATURE_CATALOG:
        values = np.asarray([row[feature] for row in rows if feature in row], dtype=float)
        if len(values):
            result[feature] = {
                "min": round(float(values.min()), 6),
                "median": round(float(np.median(values)), 6),
                "max": round(float(values.max()), 6),
            }
    return result


def build_prompt(class_name: str, support_rows: Sequence[Mapping[str, float]]) -> str:
    payload = {
        "class": class_name,
        "known_normal_support_count": len(support_rows),
        "allowed_features": compact_feature_catalog(),
        "support_feature_statistics": _support_statistics(support_rows),
    }
    return f"""
You are designing a STATIC NORMALITY rule library for few-shot industrial anomaly detection.
The attached montage contains only known-normal examples. Each row shows the RGB image on the
left and the INP-Former anomaly heatmap on the right. Never infer thresholds from defect/test
    images. Generate 8 to 16 falsifiable candidate constraints from the exact whitelist below.

Rules are constraints on normal samples:
- stable: the fact should stay near its observed normal range;
- low: normal samples should not exceed an upper envelope;
- high: normal samples should not fall below a lower envelope.

    Do not force an RGB/map quota. A later empirical verifier will reject unstable, redundant, saturated,
    or non-complementary candidates, and may keep zero rules. Treat the catalog descriptions as facts,
    not as hints about direction. Thresholds and final weights will be learned only from support-derived
    validation data, so do not invent thresholds or executable expressions. Importance and confidence
    are semantic priors only and must be non-zero numbers in [0.1, 1.0].

Return ONLY one JSON object with this schema:
{{
  "normality_summary": "one concise description of normal appearance",
  "rules": [
    {{"feature": "one allowed feature", "direction": "stable|low|high",
      "importance": 0.8, "confidence": 0.8, "reason": "short observable rationale"}}
  ]
}}

Context and measured normal facts:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first balanced JSON object, tolerating thinking/fence wrappers."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:index + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("Qwen output did not contain a valid JSON object")


def validate_proposal(payload: Mapping[str, Any], min_rules: int = 1, max_rules: int = 16) -> Dict[str, Any]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Qwen JSON must contain a rules list")
    valid = []
    seen = set()
    for item in rules:
        if not isinstance(item, Mapping):
            continue
        feature = str(item.get("feature", ""))
        if feature not in FEATURE_CATALOG or feature in seen:
            continue
        seen.add(feature)
        direction = str(item.get("direction", FEATURE_CATALOG[feature]["default_direction"])).lower()
        if direction not in {"stable", "low", "high"}:
            direction = FEATURE_CATALOG[feature]["default_direction"]
        valid.append({
            "feature": feature,
            "direction": direction,
            "importance": float(np.clip(item.get("importance", 0.7), 0.05, 1.0)),
            "confidence": float(np.clip(item.get("confidence", 0.7), 0.05, 1.0)),
            "reason": str(item.get("reason", "LLM-selected normality fact"))[:400],
        })
        if len(valid) >= max_rules:
            break
    if len(valid) < min_rules:
        raise ValueError(f"Qwen proposed only {len(valid)} valid whitelist rules")
    return {
        "normality_summary": str(payload.get("normality_summary", ""))[:1000],
        "rules": valid,
    }


class QwenRuleProposer:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "auto",
        max_new_tokens: int = 512,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = str(Path(model_path).expanduser())
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.last_raw_output = ""
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if device == "auto":
            if torch.cuda.is_available():
                target_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            else:
                target_device = "cpu"
        else:
            target_device = device
        dtype = torch.bfloat16 if target_device.startswith("cuda") else torch.float32
        load_kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype=dtype, **load_kwargs
            )
        except TypeError:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=dtype, **load_kwargs
            )
        self.model = self.model.to(target_device).eval()
        self.target_device = target_device

    def propose(
        self,
        class_name: str,
        support_rows: Sequence[Mapping[str, float]],
        montage_path: Union[str, Path],
    ) -> Dict[str, Any]:
        prompt = build_prompt(class_name, support_rows)
        json_prefix = '{"normality_summary":"'
        with Image.open(montage_path) as image:
            image = image.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json_prefix}],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        continuation = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        raw_text = json_prefix + continuation
        self.last_raw_output = raw_text
        proposal = validate_proposal(extract_json_object(raw_text))
        proposal["raw_output"] = raw_text
        return proposal
