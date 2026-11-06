#!/usr/bin/env python
"""Ask Qwen for a mask-rule *structure* without numeric parameters.

The output is validated against cross-support component groups.  Any numeric
threshold proposed by the LLM is rejected because thresholds belong to the
normal FSAD calibration stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_sam2_semantic_alignment import _extract_json
from symbolicAD.rules.mask_rule_structure import validate_mask_rule_structure


def _stable_groups(alignment):
    groups = []
    geometry = alignment["images"]
    for match in alignment["clip_cross_support_matches"]:
        left, right = int(match["image_0_mask_id"]), int(match["image_1_mask_id"])
        item_a, item_b = geometry[0][left], geometry[1][right]
        center_a, center_b = item_a["centroid_xy"], item_b["centroid_xy"]
        center_distance = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
        import math
        log_area = abs(math.log(item_a["area_ratio"] / item_b["area_ratio"]))
        if float(match["visual_cosine"]) < 0.90 or center_distance > 0.08 or log_area > 0.20:
            continue
        group_id = f"component_{len(groups):02d}"
        groups.append(
            {
                "group_id": group_id,
                "support_mask_ids": [left, right],
                "support_labels": [item_a.get("qwen_label", ""), item_b.get("qwen_label", "")],
                "centroid_xy": [
                    0.5 * (center_a[0] + center_b[0]),
                    0.5 * (center_a[1] + center_b[1]),
                ],
                "area_ratio": (item_a["area_ratio"] * item_b["area_ratio"]) ** 0.5,
                "visual_cosine": float(match["visual_cosine"]),
            }
        )
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--alignment-dir", required=True)
    parser.add_argument("--model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", default="results_mask_rule_structure")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    alignment = json.loads(Path(args.alignment_json).read_text(encoding="utf-8"))
    groups = _stable_groups(alignment)
    if len(groups) < 2:
        raise ValueError("at least two cross-support stable groups are required")
    images = [
        Image.open(Path(args.alignment_dir) / f"support_{index}_numbered_masks.png").convert("RGB")
        for index in range(2)
    ]
    group_table = json.dumps(groups, ensure_ascii=False, indent=2)
    prompt = f"""You design an EXECUTABLE PREDICATE TEMPLATE for normality rules of a PCB1 product from two NORMAL support images.
The images show SAM2 masks. Cross-support alignment has retained only these stable anonymous component groups:
{group_table}

Important constraints:
- Component IDs are authoritative. Semantic labels are noisy hints and may disagree.
- First declare every predicate signature used by rules in symbol_atoms.
- Every rule must contain an explicit predicate expression, never a family name.
- Allowed predicates are only DistToRef(m), N(component_id), and Distance(component_id,component_id).
- DistToRef and Distance rules must have type soft. N rules must have type hard.
- Every calibratable parameter must use a theta placeholder in the predicate and exactly the same key in params with value null.
- Never invent a numeric threshold, tolerance, area range, confidence, or non-null params value.
- A hard count such as N(component_00) >= 1 may contain its structural integer constant in the predicate and has params {{}}.
- semantics must say what the concrete component IDs or mask scope are required to satisfy.
- formula op must be NORMAL_IF_ALL: every predicate describes a normal condition; violating any child is anomalous.
- Relative-distance rules should be sparse, not all possible pairs.
- The normal FSAD references will determine every numeric parameter later.

Return JSON only:
{{
  "category": "pcb1",
  "source": "qwen_structure_only",
  "symbol_atoms": {{
    "DistToRef(m)": "mask m nearest-neighbour distance to the normal FSAD prototype library",
    "N(comp)": "number of query masks assigned to component comp",
    "Distance(m_i,m_j)": "Euclidean centroid distance between two masks in canonical PCB coordinates",
    "Direction(m_i,m_j)": "principal direction angle from mask i to mask j in canonical PCB coordinates"
  }},
  "rules": [
    {{
      "id": "R1",
      "type": "soft",
      "predicate": "DistToRef(m) < θ_d",
      "params": {{"θ_d": null}},
      "scope": "all_masks",
      "semantics": "Every query mask should be close to a normal PCB1 reference prototype."
    }},
    {{
      "id": "R2",
      "type": "hard",
      "predicate": "N(component_00) >= 1",
      "params": {{}},
      "scope": "component_00",
      "semantics": "component_00 should have at least one matched instance."
    }},
    {{
      "id": "R3",
      "type": "soft",
      "predicate": "Distance(component_00,component_01) < θ_gap",
      "params": {{"θ_gap": null}},
      "scope": ["component_00", "component_01"],
      "semantics": "component_00 and component_01 should preserve their normal centroid distance."
    }}
  ],
  "formula": {{
    "op": "NORMAL_IF_ALL",
    "children": ["R1", "R2", "R3"],
    "note": "All rules state normal conditions; violating any child triggers its anomaly evidence."
  }}
}}"""
    content = []
    for index, image in enumerate(images):
        copy = image.copy()
        copy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        images[index] = copy
        content.extend(
            [
                {"type": "text", "text": f"Normal support {index} numbered masks:"},
                {"type": "image", "image": copy},
            ]
        )
    content.append({"type": "text", "text": prompt})
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    model.eval()
    prefix = '{"category":"pcb1","source":"qwen_structure_only","symbol_atoms":'
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": prefix},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    inputs = processor(text=[text], images=images, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=3000, do_sample=False)
    raw = prefix + processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    parsed = _extract_json(raw)
    validated, validation = validate_mask_rule_structure(
        parsed, [item["group_id"] for item in groups]
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / f"{stamp}_pcb1_k2"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "qwen_raw.txt").write_text(raw, encoding="utf-8")
    (output_dir / "qwen_structure.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "stable_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validated_structure.json").write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    original_accepted = len(validated["rules"]) - len(validation["added"])
    if not parsed.get("rules") or original_accepted <= 0:
        raise RuntimeError(
            "Qwen produced no executable predicate rule; raw response and validation "
            f"were retained in {output_dir}"
        )
    print(json.dumps(validated, ensure_ascii=False, indent=2))
    print(f"mask rule structure written to {output_dir}")


if __name__ == "__main__":
    main()
