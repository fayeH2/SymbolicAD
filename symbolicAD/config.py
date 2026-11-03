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
    image_size: int = 224
    batch_size: int = 16
    logit_scale: float = 100.0


@dataclass
class GroundingConfig:
    alpha: float = 0.5
    component_threshold: float = 0.55
    min_instance_area: int = 8
    morphology_kernel: int = 3
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
    fail_threshold: float = 0.5
    and_epsilon: float = 1e-6


@dataclass
class FusionConfig:
    lambda_logic: float = 0.5
    beta: float = 0.5
    topk_percent: float = 1.0
    map_epsilon: float = 1e-6


@dataclass
class EvaluationConfig:
    pixel_eval_size: int = 256
    pixel_max_points: int = 4_000_000
    pro_fpr_limit: float = 0.30
    pro_thresholds: int = 64
    enable_pro: bool = True


@dataclass
class OutputConfig:
    save_maps: bool = True
    map_dtype: str = "float16"
    save_visualizations: bool = True
    save_traces: bool = True
    save_predictions: bool = True


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
    shot: int = 4
    seed: int = 668
    support_seed: int = 668
    mode: str = "baseline_logic"
    max_images_per_group: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
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
    "fusion": FusionConfig,
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
