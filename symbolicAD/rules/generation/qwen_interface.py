"""Qwen-VL interface for offline rule generation (compatible with extensible DSL)."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from PIL import Image
import torch

from symbolicAD.rules.generation.dsl import (
    NormalityRule, RuleSet, RuleType, parse_qwen_rules,
    QWEN_RULE_GENERATION_PROMPT, QWEN_RULE_VALIDATION_PROMPT,
    REGISTRY,
)
from symbolicAD.perception.scene_graph import SceneGraph


class QwenVLRuleGenerator:
    """Interface to Qwen-VL for generating normality rules from support images."""
    
    def __init__(
        self,
        model_path: str = "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking",
        device: str = "auto",
        max_new_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load Qwen-VL model (supports both Qwen2-VL and Qwen3-VL)."""
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
        ).to(self.device).eval()
    
    def _prepare_images(self, image_paths: Sequence[Union[str, Path]]) -> List[Image.Image]:
        """Load and prepare images."""
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return images
    
    def _create_montage(self, images: List[Image.Image], max_per_row: int = 2) -> Image.Image:
        """Create a contact sheet montage of images."""
        n = len(images)
        if n == 0:
            return Image.new("RGB", (448, 448), "white")
        
        # Resize all to same size
        thumb_size = (224, 224)
        thumbs = [img.resize(thumb_size, Image.Resampling.LANCZOS) for img in images]
        
        rows = (n + max_per_row - 1) // max_per_row
        montage_w = max_per_row * thumb_size[0]
        montage_h = rows * thumb_size[1]
        
        montage = Image.new("RGB", (montage_w, montage_h), "white")
        for i, thumb in enumerate(thumbs):
            row = i // max_per_row
            col = i % max_per_row
            montage.paste(thumb, (col * thumb_size[0], row * thumb_size[1]))
        
        return montage
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def generate_rules(
        self,
        category: str,
        support_image_paths: Sequence[Union[str, Path]],
        k_shot: int,
    ) -> List[NormalityRule]:
        """Generate normality rules from support images."""
        images = self._prepare_images(support_image_paths)
        montage = self._create_montage(images)
        
        prompt = QWEN_RULE_GENERATION_PROMPT.format(
            category=category, k_shot=k_shot
        )
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": montage},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[montage],
            return_tensors="pt",
        ).to(self.device)
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        # Decode only the generated part
        generated = output[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse rules
        rules = parse_qwen_rules(response, category)
        
        # Add support evidence metadata
        for rule in rules:
            rule.support_evidence = [{
                "support_image": str(p),
                "method": "qwen_generation",
            } for p in support_image_paths]
        
        return rules
    
    def validate_rule(
        self,
        rule: NormalityRule,
        image_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Validate a single rule against an image."""
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        
        prompt = QWEN_RULE_VALIDATION_PROMPT.format(
            rule_description=rule.description,
            rule_type=rule.rule_type.value,
            parameters=json.dumps(rule.to_dict(), indent=2),
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        generated = output[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"holds": False, "evidence": "parse_error", "confidence": 0.0}
    
    def batch_validate_rules(
        self,
        rules: List[NormalityRule],
        support_image_paths: Sequence[Union[str, Path]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Validate all rules against all support images."""
        results = {}
        for rule in rules:
            rule_results = []
            for img_path in support_image_paths:
                result = self.validate_rule(rule, img_path)
                rule_results.append({
                    "image": str(img_path),
                    **result,
                })
            results[rule.rule_id] = rule_results
        return results


class QwenVLRuleRefiner:
    """Refine rules using Qwen-VL feedback."""
    
    def __init__(self, generator: QwenVLRuleGenerator):
        self.generator = generator
    
    def refine_rules(
        self,
        rules: List[NormalityRule],
        support_images: Sequence[Union[str, Path]],
        scene_graphs: Sequence[Any],  # SceneGraph
    ) -> List[NormalityRule]:
        """Refine rules based on visual verification and scene graph consistency."""
        # Validate each rule on support images
        validation_results = self.generator.batch_validate_rules(rules, support_images)
        
        refined = []
        for rule in rules:
            results = validation_results.get(rule.rule_id, [])
            holds_count = sum(1 for r in results if r.get("holds", False))
            support_rate = holds_count / len(results) if results else 0.0
            
            # Update confidence based on support validation
            rule.confidence = min(rule.confidence, support_rate)
            rule.reliability = support_rate
            
            # Add validation evidence
            rule.support_evidence.extend([
                {"image": r["image"], "holds": r.get("holds"), "evidence": r.get("evidence")}
                for r in results
            ])
            
            # Keep rule if it holds on majority of supports
            if support_rate >= 0.7:
                refined.append(rule)
            else:
                print(f"Rule {rule.rule_id} rejected: support_rate={support_rate:.2f}")
        
        return refined
    
    def suggest_additional_rules(
        self,
        current_rules: List[NormalityRule],
        scene_graphs: Sequence[Any],  # SceneGraph
        category: str,
    ) -> List[NormalityRule]:
        """Ask Qwen-VL to suggest additional rules based on current rule set and scene graphs."""
        # Summarize current rules and scene graphs
        rule_summary = "\n".join([
            f"- {r.rule_id}: {r.description}" for r in current_rules
        ])
        
        scene_summary = []
        for i, graph in enumerate(scene_graphs):
            types = list(set(graph.component_types.values()))
            scene_summary.append(f"Support {i}: {len(graph.components)} components, types: {types}")
        
        prompt = f"""
Current rules for {category}:
{rule_summary}

Scene graphs from normal supports:
{chr(10).join(scene_summary)}

Are there any important normality patterns missing from the current rules?
Consider: count constraints, attribute ranges, spatial arrangements, topological connections,
compositional validity, and conditional variations.

Output additional rules in the same JSON format.
"""
        
        # Use first support image as reference
        with Image.open(support_images[0]) as img:
            reference = img.convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": reference},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.generator.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.generator.processor(
            text=[text],
            images=[reference],
            return_tensors="pt",
        ).to(self.generator.device)
        
        with torch.inference_mode():
            output = self.generator.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
        
        generated = output[:, inputs["input_ids"].shape[1]:]
        response = self.generator.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return parse_qwen_rules(response, category)