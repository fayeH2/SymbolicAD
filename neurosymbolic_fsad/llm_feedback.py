"""
LLM Feedback Integration for Evolutionary Search.

Provides Qwen-based contextual feedback for SymbolicDet-style rule evolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

from .evolutionary_search import Individual


@dataclass
class LLMFeedbackConfig:
    """Configuration for LLM feedback."""
    model_path: str = "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"
    device: str = "cuda:1"
    max_tokens: int = 1024
    temperature: float = 0.7
    feedback_interval: int = 5  # generations
    num_suggestions: int = 3


class QwenFeedbackClient:
    """Qwen client for generating evolutionary search feedback."""
    
    def __init__(self, config: LLMFeedbackConfig):
        self.config = config
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load Qwen model."""
        if self._model is None:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                import torch
                
                self._processor = AutoProcessor.from_pretrained(
                    self.config.model_path, trust_remote_code=True
                )
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=self.config.device,
                    trust_remote_code=True,
                )
                self._model.eval()
            except Exception as e:
                print(f"Failed to load Qwen model: {e}")
                self._model = None
                self._processor = None
    
    def generate_feedback(
        self,
        hall_of_fame: List[Individual],
        generation: int,
        scene_description: str,
        variable_descriptions: Dict[str, str],
        anomaly_examples: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate rule improvement suggestions from Qwen."""
        self._load_model()
        
        if self._model is None:
            return self._fallback_suggestions(hall_of_fame, variable_descriptions)
        
        try:
            import torch
            
            # Format current best rules
            rules_text = "\n".join([
                f"{i+1}. `{str(ind.expr)}` (AUROC: {ind.metrics.get('auroc', 0):.4f}, "
                f"AUPR: {ind.metrics.get('aupr', 0):.4f}, size: {ind.expr.size()})"
                for i, ind in enumerate(hall_of_fame[:5])
            ])
            
            var_desc = "\n".join([f"- {k}: {v}" for k, v in variable_descriptions.items()])
            
            prompt = f"""<|im_start|>system
You are an expert in symbolic reasoning for industrial anomaly detection.
Your task is to suggest improved logical/arithmetic rules for detecting anomalies
in a specific visual inspection scenario.

Scene: {scene_description}

Available variables:
{var_desc}

Current best rules (generation {generation}):
{rules_text}

Anomaly pattern examples: {anomaly_examples or "Not provided"}

Provide {self.config.num_suggestions} specific, executable rule suggestions.
Each suggestion must be a valid expression using the variables above and operations:
- Arithmetic: +, -, *, /, max, min
- Comparison: >, <, >=, <=, ==, != (return 1.0 if true, 0.0 if false)
- Logic: AND, OR, NOT, IMPLIES, XOR (fuzzy logic semantics)

Return ONLY a JSON array of strings, each a complete rule expression.
Example: ["(lead_count > 3) AND (lead_geometry < 0.5)", "lead_symmetry / lead_count > 1.2"]
<|im_end|>
<|im_start|>user
Suggest {self.config.num_suggestions} improved rules for the next generation.
<|im_end|>
<|im_start|>assistant
"""
            
            inputs = self._processor(
                text=prompt,
                return_tensors="pt",
                padding=True,
            ).to(self.config.device)
            
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9,
                )
            
            response = self._processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0]
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    suggestions = json.loads(response[json_start:json_end])
                    if isinstance(suggestions, list):
                        return [str(s) for s in suggestions[:self.config.num_suggestions]]
            except Exception:
                pass
            
            return self._fallback_suggestions(hall_of_fame, variable_descriptions)
            
        except Exception as e:
            print(f"Qwen feedback generation failed: {e}")
            return self._fallback_suggestions(hall_of_fame, variable_descriptions)
    
    def _fallback_suggestions(
        self,
        hall_of_fame: List[Individual],
        variable_descriptions: Dict[str, str],
    ) -> List[str]:
        """Generate heuristic suggestions when LLM unavailable."""
        suggestions = []
        var_names = list(variable_descriptions.keys())
        
        if not var_names:
            return []
        
        # Template-based suggestions
        templates = [
            # Ratios
            "{a} / {b} > 1.5",
            "{a} / {b} < 0.5",
            # Differences
            "{a} - {b} > 0.3",
            "{b} - {a} > 0.3",
            # Products
            "{a} * {b} > 0.5",
            # Logical combinations
            "({a} > 0.7) AND ({b} < 0.3)",
            "({a} > 0.5) IMPLIES ({b} < 0.5)",
            "({a} > 0.6) XOR ({b} > 0.6)",
            # Max/min
            "max({a}, {b}) > 0.8",
            "min({a}, {b}) < 0.2",
        ]
        
        for i in range(min(self.config.num_suggestions, len(templates))):
            if len(var_names) >= 2:
                a, b = var_names[i % len(var_names)], var_names[(i+1) % len(var_names)]
                suggestions.append(templates[i].format(a=a, b=b))
            else:
                a = var_names[0]
                suggestions.append(f"{a} > 0.5")
        
        return suggestions[:self.config.num_suggestions]


def create_llm_feedback_fn(
    config: LLMFeedbackConfig,
    scene_description: str,
    variable_descriptions: Dict[str, str],
    anomaly_examples: Optional[List[str]] = None,
) -> Callable[[List[Individual], int], List[str]]:
    """Create LLM feedback function for evolutionary search."""
    client = QwenFeedbackClient(config)
    
    def feedback_fn(hall_of_fame: List[Individual], generation: int) -> List[str]:
        if generation % config.feedback_interval != 0:
            return []
        return client.generate_feedback(
            hall_of_fame,
            generation,
            scene_description,
            variable_descriptions,
            anomaly_examples,
        )
    
    return feedback_fn


# Scene descriptions for MVTec categories
MVTEC_SCENE_DESCRIPTIONS = {
    "transistor": (
        "Electronic transistor component with 3 leads, package body, and PCB mounting. "
        "Normal: exactly 3 leads with symmetric spacing, straight geometry, stable package pose. "
        "Anomalies: bent leads, cut leads, misplaced leads, damaged case."
    ),
    "cable": (
        "Electrical cable with 3 inner conductors (colored wires) inside outer insulation. "
        "Normal: 3 conductors with correct color identity (brown/blue/green-yellow), "
        "proper twisted arrangement, intact outer sheath. "
        "Anomalies: missing/swapped colors, broken conductors, damaged sheath, wrong twist."
    ),
    "bottle": (
        "Glass/plastic bottle with cap and label. "
        "Normal: intact bottle body, correct cap, proper label alignment. "
        "Anomalies: broken body, missing cap, misaligned label, contamination."
    ),
    "screw": (
        "Metal screw with head and threaded shaft. "
        "Normal: intact head, complete threads, no deformation. "
        "Anomalies: stripped head, broken shaft, bent threads, contamination."
    ),
    "pill": (
        "Pharmaceutical pill/tablet. "
        "Normal: uniform shape, correct color, intact surface, proper imprint. "
        "Anomalies: chipped edges, color variation, missing imprint, contamination."
    ),
    "capsule": (
        "Gelatin capsule with two halves. "
        "Normal: both halves present, correct color, proper alignment, intact surface. "
        "Anomalies: separated halves, color mismatch, damaged surface, empty capsule."
    ),
    "metal_nut": (
        "Hexagonal metal nut. "
        "Normal: 6 symmetric sides, intact threads, no scratches. "
        "Anomalies: rounded corners, damaged threads, surface scratches, deformation."
    ),
    "zipper": (
        "Metal/plastic zipper with teeth and slider. "
        "Normal: aligned teeth, functional slider, intact tape. "
        "Anomalies: missing teeth, broken slider, separated tape, misaligned teeth."
    ),
    "hazelnut": (
        "Whole hazelnut with shell. "
        "Normal: intact shell, uniform color, no cracks. "
        "Anomalies: cracked shell, mold, discoloration, foreign object."
    ),
    "toothbrush": (
        "Plastic toothbrush with bristles and handle. "
        "Normal: complete bristle tufts, intact handle, proper alignment. "
        "Anomalies: missing bristles, broken handle, bent head, contamination."
    ),
}


def get_scene_description(category: str) -> str:
    """Get scene description for MVTec category."""
    return MVTEC_SCENE_DESCRIPTIONS.get(category, f"Industrial component: {category}")


# Default variable descriptions for transistor (from v7 logic)
TRANSISTOR_VARIABLES = {
    "lead_count": "Number of detected leads (normal=3)",
    "lead_geometry": "Lead geometry matching score (1.0=perfect match)",
    "lead_symmetry": "Lead spacing symmetry error (0.0=perfect symmetry)",
    "package_pose": "Package pose deviation from normal (0.0=stable)",
    "neural_anomaly": "INP-Former baseline anomaly score",
    "grounding": "Visual grounding confidence (lead detection quality)",
}

CABLE_VARIABLES = {
    "conductor_count": "Number of detected inner conductors (normal=3)",
    "color_identity": "Color multiset matching score (1.0=exact match)",
    "twist_regularity": "Twist pattern regularity (1.0=perfect)",
    "sheath_integrity": "Outer sheath integrity (1.0=intact)",
    "conductor_alignment": "Conductor alignment score (1.0=proper)",
    "neural_anomaly": "INP-Former baseline anomaly score",
    "grounding": "Visual grounding confidence",
}

GENERIC_VARIABLES = {
    "count": "Component count (normal=expected_count)",
    "layout": "Spatial arrangement score (1.0=normal)",
    "size": "Size consistency score (1.0=uniform)",
    "orientation": "Orientation consistency (1.0=aligned)",
    "appearance": "Appearance consistency (1.0=normal)",
    "neural_anomaly": "INP-Former baseline anomaly score",
    "grounding": "Visual grounding confidence",
}


__all__ = [
    "LLMFeedbackConfig", "QwenFeedbackClient", "create_llm_feedback_fn",
    "get_scene_description", "TRANSISTOR_VARIABLES", "CABLE_VARIABLES", "GENERIC_VARIABLES",
]