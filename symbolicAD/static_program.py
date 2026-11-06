"""Paths and strict loading for compiled, LLM-free inference programs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

from .config import Config
from .grounding.support_memory import SupportMemory, load_support_memory
from .rules.schema import RuleKB
from .symbolicdet import SegmentFirstSymbolicDet


STATIC_PROGRAM_VERSION = "symbolicad_static_v1"


@dataclass(frozen=True)
class StaticProgramPaths:
    root: Path
    rules: Path
    support_memory: Path
    symbolicdet: Path
    manifest: Path

    @classmethod
    def for_category(cls, config: Config, category: str) -> "StaticProgramPaths":
        root = config.cache_dir / "static_programs" / f"{category}_k{config.shot}"
        return cls(
            root=root,
            rules=root / "rules.json",
            support_memory=root / "support_memory",
            symbolicdet=root / "symbolicdet",
            manifest=root / "manifest.json",
        )


def load_manifest(paths: StaticProgramPaths) -> Dict[str, object]:
    if not paths.manifest.is_file():
        raise FileNotFoundError(
            f"static program is missing: {paths.manifest}. Run "
            "scripts/compile_static_program.py before inference"
        )
    payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if payload.get("version") != STATIC_PROGRAM_VERSION:
        raise RuntimeError(
            f"static program version {payload.get('version')} is incompatible; recompile"
        )
    contract = payload.get("runtime_contract")
    if not isinstance(contract, dict) or (
        contract.get("llm_allowed") is not False
        or contract.get("query_set_adaptation") is not False
    ):
        raise RuntimeError(
            "static program violates the deployment contract: inference must be "
            "LLM-free and query-set independent"
        )
    return payload


def load_rule_program(paths: StaticProgramPaths) -> RuleKB:
    load_manifest(paths)
    if not paths.rules.is_file():
        raise FileNotFoundError(paths.rules)
    return RuleKB.load(paths.rules)


def load_grounding_memory(paths: StaticProgramPaths) -> SupportMemory:
    load_manifest(paths)
    source = paths.support_memory
    legacy = source.with_suffix(".npz")
    if not source.exists() and legacy.is_file():
        source = legacy
    if not source.exists():
        raise FileNotFoundError(source)
    return load_support_memory(source)


def load_symbolicdet_program(
    paths: StaticProgramPaths, config: Config
) -> SegmentFirstSymbolicDet:
    load_manifest(paths)
    source = paths.symbolicdet
    legacy = source.with_suffix(".npz")
    if not source.exists() and legacy.is_file():
        source = legacy
    if not source.exists():
        raise FileNotFoundError(source)
    return SegmentFirstSymbolicDet.load(source, config.symbolicdet)
