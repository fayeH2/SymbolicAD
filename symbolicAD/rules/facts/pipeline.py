"""事实抽取、观测校准、反事实装配与可证伪性审计的编排层。

这一层把"图像 -> 检测 -> 场景图 -> 校准/反事实"串起来，并把所有中间量落盘，
使规则搜索可以脱离 GPU 复现：

``detect_split``
    多 GPU 并行抽取可序列化检测记录，按图像内容哈希缓存。
``calibrate_split``
    只在独立 train 正常图上估计计数/存在完整性。
``assemble_facts``
    由缓存记录重建 ``ExecutionContext``，并注入校准。
``build_counterfactual_pairs``
    对真实 normal 图像做像素干预 -> 重新检测 -> 独立确认。
``falsifiability_audit``
    逐规则检查是否存在"干预前满足、干预后违反"的已确认反事实。
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ...reasoning.predicate_executor import ExecutionContext
from ...reasoning.violation_result import RuleStatus
from ..rule_registry import RuleRegistry, default_rule_registry
from ..rule_schema import RuleSpec
from ..search.symbolicdet_evolution import CounterfactualPair
from .calibration import CountCalibration, calibrate_from_counts
from .interventions import (
    EditedImage, PixelEdit, confirm_intervention, empty_region, erase_entity,
    flip_region_vertical, relocate_entity_to_region,
    separate_entity_horizontally,
)
from .scene_builder import SceneFactBuilder, SceneFactConfig


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    """LEAKAGE-FREE split of one LOCO class around a fixed support set."""

    class_name: str
    class_root: Path
    support_paths: Tuple[Path, ...]
    calibration_paths: Tuple[Path, ...]
    validation_paths: Tuple[Path, ...]
    logical_anomaly_paths: Tuple[Path, ...] = ()
    structural_anomaly_paths: Tuple[Path, ...] = ()
    good_test_paths: Tuple[Path, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_root": str(self.class_root),
            "support_count": len(self.support_paths),
            "calibration_count": len(self.calibration_paths),
            "validation_count": len(self.validation_paths),
            "logical_anomaly_count": len(self.logical_anomaly_paths),
            "structural_anomaly_count": len(self.structural_anomaly_paths),
            "good_test_count": len(self.good_test_paths),
        }


def discover_breakfast_box(
    class_root: str | Path,
    support_paths: Sequence[str | Path],
) -> DatasetSpec:
    """Build the three disjoint normal splits plus the held-out test sets.

    ``calibration_paths`` are train/good images excluding every support whose
    bytes equal a train image; ``validation_paths`` are the ``*_val.png`` files
    excluding any byte-identical support.  A support that is not a train/good
    image at all (e.g. a re-encoded ``_val`` file) is excluded by content hash,
    not by filename.
    """
    class_root = Path(class_root)
    train_dir = class_root / "train" / "good"
    supports = tuple(Path(path).resolve() for path in support_paths)
    support_hashes = {sha256_file(path) for path in supports}

    train_images = sorted(train_dir.glob("*.png"))
    calibration = []
    validation = []
    for path in train_images:
        if sha256_file(path) in support_hashes:
            continue
        if path.stem.endswith("_val"):
            validation.append(path.resolve())
        else:
            calibration.append(path.resolve())
    if not calibration:
        raise ValueError("no calibration images left after removing supports")
    if not validation:
        raise ValueError("no independent validation images")

    def _list(subdir: str) -> Tuple[Path, ...]:
        folder = class_root / "test" / subdir
        return tuple(sorted(path.resolve() for path in folder.glob("*.png")))

    return DatasetSpec(
        class_name=class_root.name,
        class_root=class_root,
        support_paths=supports,
        calibration_paths=tuple(calibration),
        validation_paths=tuple(validation),
        logical_anomaly_paths=_list("logical_anomalies"),
        structural_anomaly_paths=_list("structural_anomalies"),
        good_test_paths=_list("good"),
    )


# ---------------------------------------------------------------------------
# Parallel detection extraction with an on-disk cache
# ---------------------------------------------------------------------------

_WORKER_BUILDER: Optional[SceneFactBuilder] = None


def _worker_init(device: str, vocabulary_payload: Optional[Dict[str, Any]]) -> None:
    global _WORKER_BUILDER
    from .open_vocab import GroundingDinoDetector, Vocabulary

    vocabulary = None
    if vocabulary_payload:
        vocabulary = Vocabulary({
            key: tuple(value) for key, value in vocabulary_payload.items()
        })
    config = SceneFactConfig(vocabulary=vocabulary) if vocabulary else SceneFactConfig()
    _WORKER_BUILDER = SceneFactBuilder(GroundingDinoDetector(device=device), config)


def _worker_extract(args: Tuple[int, str]) -> Tuple[int, Dict[str, Any]]:
    index, path = args
    if _WORKER_BUILDER is None:  # pragma: no cover - defensive
        raise RuntimeError("worker builder was not initialised")
    return index, _WORKER_BUILDER.extract(path)


def _cache_path(cache_dir: Path, path: Path) -> Path:
    return cache_dir / f"{sha256_file(path)}.json"


def detect_split(
    paths: Sequence[str | Path],
    cache_dir: str | Path,
    *,
    devices: Sequence[str] = ("cuda:0",),
    vocabulary_payload: Optional[Mapping[str, Sequence[str]]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract detection records for ``paths``, reusing the on-disk cache.

    Each device gets its own single-process pool, and those pools run
    concurrently, so every GPU holds exactly one detector for the whole run.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = [str(Path(path).resolve()) for path in paths]
    records: Dict[str, Dict[str, Any]] = {}
    missing: List[Tuple[int, str]] = []
    for index, path in enumerate(resolved):
        cached = _cache_path(cache_dir, Path(path))
        if cached.is_file():
            records[path] = json.loads(cached.read_text(encoding="utf-8"))
        else:
            missing.append((index, path))
    if not missing:
        return records

    payload = (
        {key: list(value) for key, value in vocabulary_payload.items()}
        if vocabulary_payload else None
    )
    devices = tuple(devices) or ("cuda:0",)
    chunks: List[List[Tuple[int, str]]] = [[] for _ in devices]
    for position, item in enumerate(missing):
        chunks[position % len(devices)].append(item)

    def _run_device(device: str, chunk: Sequence[Tuple[int, str]]):
        if not chunk:
            return []
        with ProcessPoolExecutor(
            max_workers=1, initializer=_worker_init, initargs=(device, payload),
        ) as pool:
            return list(pool.map(_worker_extract, chunk))

    completed = 0
    if len(devices) == 1:
        results = [_run_device(devices[0], chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as outer:
            futures = [
                outer.submit(_run_device, device, chunk)
                for device, chunk in zip(devices, chunks)
            ]
            results = [future.result() for future in futures]

    for rows in results:
        for index, record in rows:
            path = resolved[index]
            records[path] = record
            _cache_path(cache_dir, Path(path)).write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            completed += 1
            if progress:
                progress(completed, len(missing))
    return records


# ---------------------------------------------------------------------------
# Calibration and fact assembly
# ---------------------------------------------------------------------------


def calibrate_split(
    builder: SceneFactBuilder,
    records: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    explicit_entities: Iterable[str] = (),
    minimum_detection_rate: float = 0.9,
    minimum_count_stability: float = 0.9,
) -> CountCalibration:
    counts = [builder.entity_counts(record) for record in records.values()]
    return calibrate_from_counts(
        counts,
        split=split,
        source_images=list(records),
        minimum_detection_rate=minimum_detection_rate,
        minimum_count_stability=minimum_count_stability,
        explicit_entities=explicit_entities,
    )


def assemble_facts(
    builder: SceneFactBuilder,
    records: Mapping[str, Mapping[str, Any]],
    calibration: CountCalibration,
) -> Dict[str, ExecutionContext]:
    return {
        path: builder.build_from_record(record, calibration=calibration)
        for path, record in records.items()
    }


# ---------------------------------------------------------------------------
# Counterfactual assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualAuditRow:
    pair_id: str
    base_image: str
    intervention: str
    confirmed: bool
    validated: bool
    edited_sha256: str
    changed_pixels: int
    evidence: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "base_image": self.base_image,
            "intervention": self.intervention,
            "confirmed": self.confirmed,
            "validated": self.validated,
            "edited_sha256": self.edited_sha256,
            "changed_pixels": self.changed_pixels,
            "evidence": dict(self.evidence),
        }


def _regions(builder: SceneFactBuilder, record: Mapping[str, Any]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    context = builder.build_from_record(record)
    metadata = context.metadata
    container = tuple(float(value) for value in metadata["container_box"])
    divider = float(metadata["divider_x"])
    left = (container[0], container[1], divider, container[3])
    right = (divider, container[1], container[2], container[3])
    return container, left, right


def plan_interventions(
    builder: SceneFactBuilder,
    base_path: str | Path,
    record: Mapping[str, Any],
) -> Tuple[Tuple[str, Callable[[np.ndarray], Optional[EditedImage]]], ...]:
    """Concrete pixel edits for one base image, derived from its detections."""
    container, left, right = _regions(builder, record)
    plane: List[Tuple[str, Callable[[np.ndarray], Optional[EditedImage]]]] = [
        # Presence: emptying a whole compartment removes the small objects
        # without relying on a detector's residual response to a local erase.
        ("empty_left_compartment", lambda image: empty_region(image, record, left)),
        ("empty_right_compartment", lambda image: empty_region(image, record, right)),
        # Cardinality with a concrete "misplaced instance" witness.
        ("relocate_orange_left_to_right", lambda image: relocate_entity_to_region(
            image, record, "orange", left, right,
        )),
        # Vertical arrangement: mirror the packed compartment.
        ("flip_right_compartment", lambda image: _flip_with_pair(
            image, record, right, ("granola", "banana_slice"),
        )),
        ("flip_right_compartment_almond", lambda image: _flip_with_pair(
            image, record, right, ("granola", "almond"),
        )),
        # Column alignment: shift the lowest instance sideways.
        ("separate_orange_horizontally", lambda image: separate_entity_horizontally(
            image, record, "orange", left,
        )),
    ]
    return tuple(plane)


def _flip_with_pair(
    image: np.ndarray,
    record: Mapping[str, Any],
    region: Sequence[float],
    pair: Tuple[str, str],
) -> Optional[EditedImage]:
    edited = flip_region_vertical(image, record, region)
    if edited is None:
        return None
    edit = PixelEdit(
        kind=edited.edit.kind,
        description=edited.edit.description,
        boxes=edited.edit.boxes,
        source_entity=edited.edit.source_entity,
        target_entity=edited.edit.target_entity,
        extra={"region": list(region), "comparison_pair": list(pair)},
    )
    return EditedImage(edited.image, edit, edited.changed_pixels, edited.sha256)


def materialize_interventions(
    builder: SceneFactBuilder,
    base_paths: Sequence[str | Path],
    records: Mapping[str, Mapping[str, Any]],
    edited_dir: str | Path,
) -> Dict[str, Dict[str, Any]]:
    """Write edited images to disk and return ``{edited_path: edit_metadata}``."""
    edited_dir = Path(edited_dir)
    edited_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, Any]] = {}
    for base in base_paths:
        base = str(Path(base).resolve())
        record = records[base]
        image = np.asarray(Image.open(base).convert("RGB"))
        for name, apply in plan_interventions(builder, base, record):
            edited = apply(image)
            if edited is None:
                continue
            target = edited_dir / f"{Path(base).stem}__{name}.png"
            Image.fromarray(edited.image).save(target)
            manifest[str(target.resolve())] = {
                "base_image": base,
                "intervention": name,
                "edit": edited.to_dict(),
            }
    return manifest


def finalize_counterfactual_pairs(
    builder: SceneFactBuilder,
    records: Mapping[str, Mapping[str, Any]],
    edited_records: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    calibration: CountCalibration,
) -> Tuple[Tuple[CounterfactualPair, ...], Tuple[CounterfactualAuditRow, ...]]:
    pairs: List[CounterfactualPair] = []
    audit: List[CounterfactualAuditRow] = []
    for edited_path, entry in sorted(manifest.items()):
        base_path = entry["base_image"]
        edit_payload = entry["edit"]
        edit = PixelEdit(
            kind=edit_payload["edit"]["kind"],
            description=edit_payload["edit"]["description"],
            boxes=tuple(tuple(row) for row in edit_payload["edit"]["boxes"]),
            source_entity=edit_payload["edit"]["source_entity"],
            target_entity=edit_payload["edit"]["target_entity"],
            extra=dict(edit_payload["edit"]["extra"]),
        )
        before_record = records[base_path]
        after_record = edited_records.get(edited_path)
        if after_record is None:
            audit.append(CounterfactualAuditRow(
                pair_id=f"{Path(base_path).stem}::{entry['intervention']}",
                base_image=base_path, intervention=entry["intervention"],
                confirmed=False, validated=False,
                edited_sha256=edit_payload["sha256"], changed_pixels=1,
                evidence={"reason": "edited record missing"},
            ))
            continue
        confirmed, evidence = confirm_intervention(before_record, after_record, edit)
        before_context = builder.build_from_record(before_record, calibration=calibration)
        after_context = builder.build_from_record(
            after_record, calibration=calibration, edited=True,
            intervention={"kind": edit.kind, "description": edit.description},
        )
        pair_id = f"{Path(base_path).stem}::{entry['intervention']}"
        row = CounterfactualAuditRow(
            pair_id=pair_id, base_image=base_path,
            intervention=entry["intervention"], confirmed=bool(confirmed),
            validated=bool(confirmed),
            edited_sha256=edit_payload["sha256"],
            changed_pixels=int(edit_payload["changed_pixels"]),
            evidence=evidence,
        )
        audit.append(row)
        if confirmed:
            pairs.append(CounterfactualPair(
                pair_id=pair_id,
                normal=before_context,
                counterfactual=after_context,
                intervention=entry["intervention"],
                validated=True,
                validation_evidence={
                    "edited_sha256": edit_payload["sha256"],
                    "changed_pixels": edit_payload["changed_pixels"],
                    "base_image": base_path,
                    "edited_image": edited_path,
                    "measurements": dict(evidence),
                },
            ))
    return tuple(pairs), tuple(audit)


# ---------------------------------------------------------------------------
# Falsifiability audit
# ---------------------------------------------------------------------------


def falsifiability_audit(
    rules: Sequence[RuleSpec],
    pairs: Sequence[CounterfactualPair],
    *,
    registry: Optional[RuleRegistry] = None,
) -> Tuple[Dict[str, Any], ...]:
    """For each rule, ask whether any confirmed pair turns SATISFIED -> VIOLATED.

    A rule that no intervention can falsify is tautological *with respect to the
    perception contract*: including it cannot reduce ``L_cf`` and it cannot
    catch an anomaly, so the report marks it explicitly.
    """
    registry = registry or default_rule_registry()
    rows = []
    for rule in rules:
        capability = registry.resolve(rule.canonical_predicate)
        if capability is None:
            rows.append({
                "rule_id": rule.rule_id, "predicate": rule.canonical_predicate,
                "executable": False, "falsified_by": [], "unknown_pairs": 0,
                "falsifiable": False,
            })
            continue
        falsified_by: List[str] = []
        unknown = 0
        max_margin = 0.0
        for pair in pairs:
            before = capability.executor.execute(rule, pair.normal)
            after = capability.executor.execute(rule, pair.counterfactual)
            if before.status == RuleStatus.UNKNOWN or after.status == RuleStatus.UNKNOWN:
                unknown += 1
                continue
            margin = float(after.violation_score) - float(before.violation_score)
            max_margin = max(max_margin, margin)
            if (
                before.status == RuleStatus.SATISFIED
                and after.status == RuleStatus.VIOLATED
            ):
                falsified_by.append(pair.pair_id)
        rows.append({
            "rule_id": rule.rule_id,
            "predicate": rule.canonical_predicate,
            "executable": True,
            "falsified_by": falsified_by,
            "unknown_pairs": unknown,
            "max_margin": max_margin,
            "falsifiable": bool(falsified_by),
        })
    return tuple(rows)


__all__ = [
    "DatasetSpec", "discover_breakfast_box", "sha256_file", "detect_split",
    "calibrate_split", "assemble_facts", "plan_interventions",
    "materialize_interventions", "finalize_counterfactual_pairs",
    "falsifiability_audit", "CounterfactualAuditRow",
]
