"""Typed configuration objects for the staged SymbolicAD pipeline.

All numeric constants used by the pipeline live here (or in ``configs/default.yaml``)
so that they never leak into individual modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    clip_model: str = "hf-hub:timm/vit_base_patch16_clip_224.openai"
    device: str = "cuda:0"
    image_size: int = 448
    batch_size: int = 4
    logit_scale: float = 100.0
    baseline: str = "clip_adaptive"
    knn: int = 1
    max_bank_size: int = 0
    spatial_radius: int = 1
    spatial_weight: float = 0.5
    # Dense multi-layer features.  PromptAD/AnomalyCLIP show that middle ViT
    # blocks are much better for localisation than the final projected layer.
    # Indices are zero-based over the vision transformer blocks and are capped
    # automatically for other CLIP backbones.
    feature_layers: Tuple[int, ...] = (3, 8, 11)
    layer_fusion: str = "mean"
    local_radius: int = 1
    local_weight: float = 0.35
    whiten: bool = False
    whiten_components: int = 128
    whiten_eps: float = 1.0e-3
    # Textual (normal-vs-damaged prompt) branch.  Fused harmonically with the
    # visual memory branch as in PromptAD.
    text_visual_weight: float = 0.0
    text_visual_fusion: str = "harmonic"
    foreground_masking: bool = True
    foreground_soft: float = 0.35
    image_score_percent: float = 1.0
    image_text_weight: float = 0.1
    image_global_weight: float = 0.0
    # Fixed by support-only protocol. ``auto`` is a transductive ablation
    # because it inspects the query score distribution and therefore requires
    # ``evaluation.allow_transductive`` to be enabled explicitly.
    feature_source: str = "both"
    projected_weight: float = 0.25
    auto_select_samples: int = 120
    auto_margin: float = 0.08
    normalise_layers: bool = False


@dataclass
class GroundingConfig:
    alpha: float = 0.5
    component_threshold: float = 0.55
    min_instance_area: int = 2
    morphology_kernel: int = 1
    support_quantile_low: float = 0.50
    support_quantile_high: float = 0.99
    max_prototypes_per_component: int = 32
    text_embedding_cache: bool = True


@dataclass
class LogicConfig:
    count_sigma: float = 1.0
    ratio_sigma: float = 0.25
    lambda_feature: float = 0.3
    lambda_relation: float = 0.5
    lambda_pair_prompt: float = 0.2
    dummy_match_cost: float = 0.5
    spatial_temperature: float = 1.0
    eta: float = 0.5
    rescore_temperature: float = 0.07
    color_sigma: float = 0.18
    fail_threshold: float = 0.5
    and_epsilon: float = 1e-6
    rescore_enabled: bool = False
    # Convert raw atom truths into support-relative normalities.  An atom whose
    # support truth is already low is a grounding failure, not an anomaly cue.
    support_calibration: bool = True
    min_atom_support_truth: float = 0.55
    heatmap_sigma: float = 0.8
    soft_responsibility: bool = True
    # Floor the support gap so two nearly identical support images do not turn
    # tiny CLIP score fluctuations into full-scale violations.
    calibration_min_gap: float = 0.10
    attribute_min_gap: float = 0.05


@dataclass
class QwenConfig:
    enabled: bool = True
    model_path: str = "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"
    device: str = "cuda:1"
    max_new_tokens: int = 3000
    max_image_side: int = 768
    cache_subdir: str = "qwen_rules"
    allow_template_fallback: bool = False


@dataclass
class CompilerConfig:
    enabled: bool = True
    deduplicate: bool = True
    drop_implied_exist: bool = True
    drop_redundant_one_to_one: bool = True
    repair_counts: bool = False
    repair_relation_direction: bool = False
    add_missing_counts: bool = False
    disable_inconsistent_atoms: bool = True
    stable_count_std: float = 0.5
    count_in_range_sigma: float = 1.0
    max_count_range_width: int = 6
    min_attribute_support_score: float = 0.05


@dataclass
class FusionConfig:
    lambda_logic: float = 0.25
    beta: float = 0.0
    topk_percent: float = 1.0
    map_epsilon: float = 1e-6
    # Scale the configured fusion weights by support-estimated rule reliability.
    # A rule bank that already fails on normal support must abstain instead of
    # painting normal images with false anomalies.
    auto_from_support: bool = True
    min_reliability: float = 0.55
    min_coverage: float = 0.35
    # Leave-one-out support logic maps must be quiet before pixel fusion is
    # allowed.  Noisy responsibility maps cannot be rescued by a small lambda.
    max_support_logic_noise: float = 0.12
    # Small image-level symbolic residual.  Pixel fusion is gated separately;
    # this term only shifts the image score and cannot damage the heat-map.
    image_logic_weight: float = 0.02
    image_logic_scale: float = 1.0
    max_support_image_noise: float = 0.30


@dataclass
class SymbolicDetConfig:
    """Support-only segmentation, dynamic expressions and sparse MoE."""

    enabled: bool = True
    analysis_size: int = 128
    color_clusters: int = 4
    grid_rows: int = 4
    grid_cols: int = 4
    min_segment_area_ratio: float = 0.0015
    max_segments: int = 48
    foreground_color_distance: float = 0.08
    morphology_kernel: int = 3
    relative_margin: float = 0.15
    occupancy_margin: float = 0.04
    color_margin: float = 0.06
    position_margin: float = 0.06
    count_margin: float = 0.5
    violation_temperature: float = 1.0
    expert_top_k: int = 2
    appearance_weight: float = 0.40
    structure_weight: float = 0.35
    relation_weight: float = 0.25
    fusion_weight: float = 0.25
    image_weight: float = 0.05
    min_expression_confidence: float = 0.10


@dataclass
class EvaluationConfig:
    pixel_eval_size: int = 256
    pixel_max_points: int = 4_000_000
    pro_fpr_limit: float = 0.30
    pro_thresholds: int = 64
    enable_pro: bool = True
    # Strict FSAD never adapts model/rule choices on the test query set. This
    # opt-in switch exists only for clearly labelled transductive ablations.
    allow_transductive: bool = False


@dataclass
class OutputConfig:
    save_maps: bool = True
    map_dtype: str = "float16"
    save_visualizations: bool = True
    save_traces: bool = True
    save_predictions: bool = True
    timestamped: bool = True
    method_name: str = "SymbolicAD"
    image_size: int = 400
    save_reference_images: bool = True


@dataclass
class PathsConfig:
    data_root: str = "/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO"
    cache_dir: str = "cache"
    results_dir: str = "results"
    clip_cache_dir: str = "/home/hjh/.cache/huggingface/hub"


@dataclass
class Config:
    dataset: str = "mvtec_loco"
    category: str = ""
    shot: int = 2
    seed: int = 668
    support_seed: int = 668
    mode: str = "baseline_logic"
    max_images_per_group: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    symbolicdet: SymbolicDetConfig = field(default_factory=SymbolicDetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def cache_dir(self) -> Path:
        return Path(self.paths.cache_dir).expanduser()

    @property
    def results_dir(self) -> Path:
        return Path(self.paths.results_dir).expanduser()

    @property
    def data_root(self) -> Path:
        return Path(self.paths.data_root).expanduser()

    @property
    def rule_cache_dir(self) -> Path:
        return self.cache_dir / "rules"

    @property
    def support_memory_dir(self) -> Path:
        return self.cache_dir / "support_memory"

    @property
    def text_embedding_dir(self) -> Path:
        return self.cache_dir / "text_embeddings"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "grounding": GroundingConfig,
    "logic": LogicConfig,
    "qwen": QwenConfig,
    "compiler": CompilerConfig,
    "fusion": FusionConfig,
    "symbolicdet": SymbolicDetConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
    "paths": PathsConfig,
}


def _build_section(cls, values: Mapping[str, Any]):
    known = {item.name for item in fields(cls)}
    filtered = {key: value for key, value in values.items() if key in known}
    return cls(**filtered)


def _flatten(values: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in values.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


def _apply_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """Apply dotted ``section.field`` overrides to a config object."""
    for dotted, value in _flatten(overrides).items():
        if "." in dotted:
            section_name, field_name = dotted.rsplit(".", 1)
            section = getattr(config, section_name, None)
            if section is None:
                continue
            if hasattr(section, field_name):
                setattr(section, field_name, value)
        elif hasattr(config, dotted):
            setattr(config, dotted, value)
    return config


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """Load YAML/JSON config and optionally apply dotted overrides."""
    values: Dict[str, Any] = {}
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            import json

            values = json.loads(path.read_text(encoding="utf-8"))

    scalar = {
        key: value
        for key, value in values.items()
        if key not in _SECTIONS and not isinstance(value, Mapping)
    }
    config = Config(**{key: value for key, value in scalar.items() if hasattr(Config, key)})
    for section_name, section_cls in _SECTIONS.items():
        section_values = values.get(section_name, {})
        setattr(config, section_name, _build_section(section_cls, section_values))

    if overrides:
        config = _apply_overrides(config, overrides)
    return config


def resolve_support_batch_size(config: Config) -> int:
    return max(1, int(config.model.batch_size))


def map_grid(config: Config) -> Tuple[int, int]:
    patch = 16
    side = max(1, int(config.model.image_size) // patch)
    return side, side
