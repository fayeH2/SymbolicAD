#!/usr/bin/env python3
"""Independent validator for the v3 relational FSAD cable experiment.

This program intentionally does not import metric helpers from the experiment.
It recomputes every reported test metric from the persisted prediction table,
audits the train/test and calibration lineage, verifies cryptographic
provenance, and exercises the compiled rule engine with all LLM imports
explicitly blocked.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


BOOTSTRAP_SEED = 20261013
BOOTSTRAP_ITERATIONS = 2000
SCHEMA_VERSION = "3.0-relational-validation"
NORMAL_TYPES = {"good", "ok"}
REQUIRED_FILES = (
    "metrics_v3_relational.json",
    "predictions_v3_relational.csv",
    "split_v3_relational.json",
    "normality_rules_v3_relational.json",
    "relational_calibration_manifest.json",
    "crossfit_manifest_v3_relational.json",
    "qwen_relational_proposal.json",
    "qwen_relational_raw.txt",
    "normal_support_relational_montage.jpg",
)
SCORE_COLUMNS = {
    "baseline": "baseline_score",
    "relational": "relational_score",
    "fused": "decision_score",
}
SAVED_METRIC_GROUPS = {
    "baseline": "baseline",
    "relational": "relations_only",
    "fused": "baseline_plus_relations",
}


class ValidationError(ValueError):
    """Raised when persisted artifacts cannot support a requested check."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve(strict=False))


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(root: Path) -> Dict[str, Any]:
    """Fully hash a local model directory, including every weight shard."""
    if not root.is_dir():
        raise ValidationError(f"Qwen model directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValidationError(f"Qwen model directory contains no files: {root}")
    aggregate = hashlib.sha256()
    components: Dict[str, Dict[str, Any]] = {}
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _file_sha256(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        total_bytes += size
        components[relative] = {"size_bytes": int(size), "sha256": digest}

    index_path = root / "model.safetensors.index.json"
    indexed_shards: List[str] = []
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map", {}) if isinstance(index, Mapping) else {}
        if isinstance(weight_map, Mapping):
            indexed_shards = sorted({str(value) for value in weight_map.values()})
    hashed_names = set(components)
    return {
        "path": str(root),
        "aggregate_sha256": aggregate.hexdigest(),
        "algorithm": "sha256(relative_path NUL size NUL file_sha256) over sorted files",
        "file_count": len(files),
        "total_bytes": int(total_bytes),
        "indexed_weight_shards": indexed_shards,
        "all_indexed_weight_shards_hashed": bool(indexed_shards)
        and set(indexed_shards).issubset(hashed_names),
        "components": components,
    }


def _safe_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValidationError("labels and scores must be one-dimensional")
    if len(labels_array) != len(scores_array) or len(labels_array) == 0:
        raise ValidationError("labels and scores must be non-empty and have equal length")
    if set(np.unique(labels_array).tolist()) != {0, 1}:
        raise ValidationError("metric population must contain both labels {0, 1}")
    if not np.isfinite(scores_array).all():
        raise ValidationError("scores contain NaN or infinity")
    auroc = (
        0.5
        if len(np.unique(scores_array)) < 2
        else float(roc_auc_score(labels_array, scores_array))
    )
    return {
        "i_auroc": auroc,
        "i_aupr": float(average_precision_score(labels_array, scores_array)),
    }


def _stratified_paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    metrics: Sequence[str] = ("i_auroc", "i_aupr"),
) -> Dict[str, Any]:
    """Paired percentile bootstrap, resampling each binary class separately."""
    if iterations != BOOTSTRAP_ITERATIONS:
        raise ValidationError(
            f"registered validation requires exactly {BOOTSTRAP_ITERATIONS} iterations"
        )
    labels = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline, dtype=float)
    candidate_arrays = {
        name: np.asarray(values, dtype=float) for name, values in candidates.items()
    }
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    if len(normal) == 0 or len(anomaly) == 0:
        raise ValidationError("stratified bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    draws = {
        name: {metric: np.empty(iterations, dtype=float) for metric in metrics}
        for name in candidate_arrays
    }
    for iteration in range(iterations):
        chosen = np.concatenate((
            rng.choice(normal, size=len(normal), replace=True),
            rng.choice(anomaly, size=len(anomaly), replace=True),
        ))
        sampled_labels = labels[chosen]
        baseline_metrics = _safe_metrics(sampled_labels, baseline[chosen])
        for name, values in candidate_arrays.items():
            candidate_metrics = _safe_metrics(sampled_labels, values[chosen])
            for metric in metrics:
                draws[name][metric][iteration] = (
                    candidate_metrics[metric] - baseline_metrics[metric]
                )

    result: Dict[str, Any] = {
        "iterations": int(iterations),
        "seed": int(seed),
        "confidence_level": 0.95,
        "method": "stratified paired percentile bootstrap with replacement",
    }
    for name, metric_draws in draws.items():
        result[f"{name}_minus_baseline"] = {}
        for metric, values in metric_draws.items():
            result[f"{name}_minus_baseline"][f"delta_{metric}_ci95"] = [
                float(value) for value in np.quantile(values, [0.025, 0.975])
            ]
            result[f"{name}_minus_baseline"][f"probability_delta_{metric}_gt_0"] = (
                float(np.mean(values > 0.0))
            )
    return result


def _read_predictions(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "defect_type", "label", *SCORE_COLUMNS.values()}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValidationError(f"prediction CSV missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValidationError("prediction CSV is empty")
    try:
        arrays = {
            name: np.asarray([float(row[column]) for row in rows], dtype=float)
            for name, column in SCORE_COLUMNS.items()
        }
        arrays["labels"] = np.asarray([int(row["label"]) for row in rows], dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid numeric value in prediction CSV: {exc}") from exc
    return rows, arrays


def _metrics_by_scope(
    rows: Sequence[Mapping[str, str]], arrays: Mapping[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    labels = arrays["labels"]
    defect_types = np.asarray([str(row["defect_type"]) for row in rows])
    output: Dict[str, Dict[str, Any]] = {}

    def calculate(scope: str, mask: np.ndarray, defect_type: str) -> None:
        scoped_labels = labels[mask]
        metrics = {
            name: _safe_metrics(scoped_labels, values[mask])
            for name, values in arrays.items()
            if name in SCORE_COLUMNS
        }
        output[scope] = {
            "scope": scope,
            "defect_type": defect_type,
            "normal_count": int((scoped_labels == 0).sum()),
            "anomaly_count": int((scoped_labels == 1).sum()),
            "metrics": metrics,
            "fused_minus_baseline": {
                "delta_i_auroc": (
                    metrics["fused"]["i_auroc"] - metrics["baseline"]["i_auroc"]
                ),
                "delta_i_aupr": (
                    metrics["fused"]["i_aupr"] - metrics["baseline"]["i_aupr"]
                ),
            },
            "relational_minus_baseline": {
                "delta_i_auroc": (
                    metrics["relational"]["i_auroc"]
                    - metrics["baseline"]["i_auroc"]
                ),
                "delta_i_aupr": (
                    metrics["relational"]["i_aupr"] - metrics["baseline"]["i_aupr"]
                ),
            },
            "relation_trigger_rate_anomaly": float(
                np.mean(arrays["relational"][mask][scoped_labels == 1] > 0.0)
            ),
        }

    calculate("overall", np.ones(len(rows), dtype=bool), "all_anomalies")
    for defect_type in sorted(set(defect_types) - NORMAL_TYPES):
        mask = np.isin(defect_types, [*NORMAL_TYPES, defect_type])
        calculate(f"{defect_type}_vs_good", mask, defect_type)
    return output


def _walk_strings(value: Any, location: str = "$") -> Iterable[Tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_strings(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{location}[{index}]")


def _path_references(value: Any, targets: Sequence[str]) -> List[Dict[str, str]]:
    resolved_targets = {_normalise_path(target) for target in targets}
    raw_targets = {str(target) for target in targets}
    matches = []
    for location, text in _walk_strings(value):
        direct = text in raw_targets
        embedded = any(target and target in text for target in raw_targets)
        resolved = False
        if "://" not in text and (Path(text).is_absolute() or "/" in text):
            resolved = _normalise_path(text) in resolved_targets
        if direct or embedded or resolved:
            matches.append({"location": location, "value": text})
    return matches


def _source_runtime_audit(
    repo_root: Path, rule_library: Path, probe_image: Path
) -> Dict[str, Any]:
    """Static import audit plus a subprocess run with LLM imports denied."""
    source_paths = [
        repo_root / "neurosymbolic_fsad" / "relational_inference.py",
        repo_root / "neurosymbolic_fsad" / "cable_relations.py",
        repo_root / "neurosymbolic_fsad" / "inpformer_adapter.py",
    ]
    forbidden_prefixes = (
        "transformers",
        "neurosymbolic_fsad.qwen_relations",
        "neurosymbolic_fsad.qwen_rules",
    )
    imported_modules: Dict[str, List[str]] = {}
    forbidden_imports: List[Dict[str, str]] = []
    pretrained_calls: List[Dict[str, Any]] = []
    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = "." * node.level + module
                names.append(module)
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Attribute) and function.attr == "from_pretrained":
                    pretrained_calls.append({"file": str(path), "line": node.lineno})
        imported_modules[str(path.relative_to(repo_root))] = names
        for name in names:
            canonical = name.lstrip(".")
            if any(
                canonical == prefix or canonical.startswith(prefix + ".")
                for prefix in forbidden_prefixes
            ):
                forbidden_imports.append({"file": str(path), "module": name})

    dynamic_code = r'''
import importlib.abc
import json
import sys

class DenyLLM(importlib.abc.MetaPathFinder):
    blocked = ("transformers", "neurosymbolic_fsad.qwen_relations", "neurosymbolic_fsad.qwen_rules")
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + ".") for item in self.blocked):
            raise ImportError("validator intentionally blocked LLM module: " + fullname)
        return None

sys.meta_path.insert(0, DenyLLM())
import neurosymbolic_fsad.relational_inference
from neurosymbolic_fsad.cable_relations import CableRelationalRuleEngine
engine = CableRelationalRuleEngine.load(sys.argv[1])
score, evidence, facts, scene = engine.evaluate(sys.argv[2])
loaded = sorted(name for name in sys.modules if (
    name == "transformers" or name.startswith("transformers.")
    or name == "neurosymbolic_fsad.qwen_relations"
    or name == "neurosymbolic_fsad.qwen_rules"
))
print(json.dumps({
    "relational_score": score,
    "evidence_count": len(evidence),
    "fact_count": len(facts),
    "loaded_forbidden_modules": loaded,
}))
'''
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", dynamic_code, str(rule_library), str(probe_image)],
        cwd=str(repo_root),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    dynamic_payload: Dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            dynamic_payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            dynamic_payload = {}
    return {
        "source_files": [str(path) for path in source_paths],
        "imports_by_file": imported_modules,
        "forbidden_imports": forbidden_imports,
        "from_pretrained_calls": pretrained_calls,
        "static_runtime_has_no_llm_dependency": (
            not forbidden_imports and not pretrained_calls
        ),
        "blocked_llm_subprocess": {
            "returncode": int(completed.returncode),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "payload": dynamic_payload,
        },
        "compiled_rules_run_with_llm_blocked": bool(
            completed.returncode == 0
            and dynamic_payload
            and not dynamic_payload.get("loaded_forbidden_modules")
        ),
    }


def _experiment_source_audit(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "neurosymbolic_fsad" / "experiment_relational.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    run_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ),
        None,
    )
    if run_node is None:
        raise ValidationError("experiment_relational.py does not define run()")
    calibration_calls = []
    library_call_lines = []
    label_assignment_lines = []
    # Restrict this ordering check to run().  Helper functions also use a local
    # variable named ``labels`` and are irrelevant to induction-time lineage.
    for node in ast.walk(run_node):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == "build_relational_calibration_views":
                first_arg = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else ""
                calibration_calls.append({"line": node.lineno, "first_argument": first_arg})
            elif name == "build_relational_library":
                library_call_lines.append(node.lineno)
        elif isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "labels" in targets:
                label_assignment_lines.append(node.lineno)
    return {
        "path": str(path),
        "calibration_calls": calibration_calls,
        "relational_calibration_called_with_support_paths": bool(calibration_calls)
        and all(item["first_argument"] == "support_paths" for item in calibration_calls),
        "library_call_lines": library_call_lines,
        "test_label_assignment_lines": label_assignment_lines,
        "library_compiled_before_test_label_array": bool(
            library_call_lines
            and label_assignment_lines
            and max(library_call_lines) < min(label_assignment_lines)
        ),
    }


def _proposal_sha256(proposal: Mapping[str, Any]) -> str:
    # This reproduces experiment_relational.py exactly (default separators and
    # ensure_ascii=True, with sorted object keys).
    text = json.dumps(dict(proposal), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _saved_metric_errors(
    saved: Mapping[str, Any], scopes: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    errors: Dict[str, float] = {}
    overall = scopes["overall"]
    for local_name, saved_name in SAVED_METRIC_GROUPS.items():
        saved_group = saved.get(saved_name)
        if not isinstance(saved_group, Mapping):
            raise ValidationError(f"saved metrics missing group: {saved_name}")
        for metric in ("i_auroc", "i_aupr"):
            errors[f"{saved_name}.{metric}"] = abs(
                float(saved_group[metric])
                - float(overall["metrics"][local_name][metric])
            )
    for metric, key in (("i_auroc", "delta_i_auroc"), ("i_aupr", "delta_i_aupr")):
        errors[key] = abs(
            float(saved[key]) - float(overall["fused_minus_baseline"][f"delta_{metric}"])
        )

    saved_per_type = saved.get("per_defect_type", {})
    if not isinstance(saved_per_type, Mapping):
        raise ValidationError("saved per_defect_type must be an object")
    for scope_name, result in scopes.items():
        if scope_name == "overall":
            continue
        defect_type = str(result["defect_type"])
        stored = saved_per_type.get(defect_type)
        if not isinstance(stored, Mapping):
            raise ValidationError(f"saved metrics missing defect type: {defect_type}")
        for local_name, saved_key in (
            ("baseline", "baseline_i_auroc"),
            ("relational", "relational_i_auroc"),
            ("fused", "fused_i_auroc"),
        ):
            errors[f"per_defect_type.{defect_type}.{saved_key}"] = abs(
                float(stored[saved_key])
                - float(result["metrics"][local_name]["i_auroc"])
            )
    return {
        "absolute_errors": errors,
        "maximum_absolute_error": float(max(errors.values(), default=0.0)),
    }


def validate_run(
    run_dir: Path,
    output_root: Path,
    metric_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve(strict=False)
    output_root = output_root.expanduser().resolve(strict=False)
    repo_root = Path(__file__).resolve().parents[1]
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValidationError(f"required artifacts missing: {missing}")

    saved = _read_json(run_dir / "metrics_v3_relational.json")
    split = _read_json(run_dir / "split_v3_relational.json")
    library = _read_json(run_dir / "normality_rules_v3_relational.json")
    manifest = _read_json(run_dir / "relational_calibration_manifest.json")
    crossfit_manifest = _read_json(run_dir / "crossfit_manifest_v3_relational.json")
    proposal = _read_json(run_dir / "qwen_relational_proposal.json")
    for name, value in (("metrics", saved), ("split", split), ("library", library),
                        ("proposal", proposal)):
        if not isinstance(value, Mapping):
            raise ValidationError(f"{name} artifact must be a JSON object")
    if not isinstance(manifest, list) or not manifest or not all(
        isinstance(row, Mapping) for row in manifest
    ):
        raise ValidationError("relational calibration manifest must be a non-empty list")
    if not isinstance(crossfit_manifest, list) or not all(
        isinstance(row, Mapping) for row in crossfit_manifest
    ):
        raise ValidationError("crossfit manifest must be a list of fold objects")

    prediction_rows, arrays = _read_predictions(
        run_dir / "predictions_v3_relational.csv"
    )
    labels = arrays["labels"]
    scopes = _metrics_by_scope(prediction_rows, arrays)
    saved_errors = _saved_metric_errors(saved, scopes)

    support_raw = split.get("support")
    test_raw = split.get("test")
    if not isinstance(support_raw, list) or not isinstance(test_raw, list):
        raise ValidationError("split artifact must contain support and test lists")
    if not all(isinstance(item, Mapping) and "path" in item for item in test_raw):
        raise ValidationError("split test entries must be objects containing path")
    support_paths = [_normalise_path(value) for value in support_raw]
    test_paths = [_normalise_path(item["path"]) for item in test_raw]
    prediction_paths = [_normalise_path(row["path"]) for row in prediction_rows]
    library_support_raw = library.get("support_paths", [])
    if not isinstance(library_support_raw, list):
        raise ValidationError("rule library support_paths must be a list")
    library_support = [_normalise_path(value) for value in library_support_raw]

    split_by_path = {_normalise_path(item["path"]): item for item in test_raw}
    predictions_match_split = len(prediction_paths) == len(test_paths) and set(
        prediction_paths
    ) == set(test_paths)
    labels_match_split = predictions_match_split and all(
        int(row["label"]) == int(split_by_path[path]["label"])
        for row, path in zip(prediction_rows, prediction_paths)
    )
    types_match_split = predictions_match_split and all(
        str(row["defect_type"]) == str(split_by_path[path].get("defect_type", ""))
        for row, path in zip(prediction_rows, prediction_paths)
    )
    labels_match_paths = all(
        int(row["label"])
        == (0 if str(row["defect_type"]).lower() in NORMAL_TYPES else 1)
        for row in prediction_rows
    )

    provenance = library.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValidationError("rule-library provenance must be an object")
    checkpoint_path = Path(str(provenance.get("checkpoint", ""))).expanduser().resolve(
        strict=False
    )
    checkpoint_actual_hash = (
        _file_sha256(checkpoint_path) if checkpoint_path.is_file() else ""
    )
    checkpoint_expected_hash = str(provenance.get("checkpoint_sha256", ""))

    expected_support_hashes_raw = provenance.get("support_sha256", {})
    if not isinstance(expected_support_hashes_raw, Mapping):
        expected_support_hashes_raw = {}
    expected_support_hashes = {
        _normalise_path(path): str(digest)
        for path, digest in expected_support_hashes_raw.items()
    }
    actual_support_hashes = {
        path: _file_sha256(Path(path)) if Path(path).is_file() else ""
        for path in support_paths
    }
    support_hash_rows = [
        {
            "path": path,
            "expected_sha256": expected_support_hashes.get(path, ""),
            "actual_sha256": actual_support_hashes[path],
            "matches": bool(
                expected_support_hashes.get(path)
                and expected_support_hashes[path] == actual_support_hashes[path]
            ),
        }
        for path in support_paths
    ]

    proposal_actual_hash = _proposal_sha256(proposal)
    proposal_expected_hash = str(provenance.get("proposal_sha256", ""))
    proposal_artifact_hash = _file_sha256(run_dir / "qwen_relational_proposal.json")
    raw_output_hash = _file_sha256(run_dir / "qwen_relational_raw.txt")
    montage_hash = _file_sha256(run_dir / "normal_support_relational_montage.jpg")
    qwen_model_from_proposal = _normalise_path(proposal.get("model", ""))
    qwen_model_from_library = _normalise_path(library.get("qwen", {}).get("model", ""))
    qwen_model_path = Path(qwen_model_from_library)
    qwen_model_fingerprint = _directory_sha256(qwen_model_path)
    recorded_model_hash = str(
        provenance.get("qwen_model_aggregate_sha256")
        or provenance.get("qwen_model_sha256")
        or provenance.get("llm_model_sha256")
        or ""
    )

    crossfit_held = [_normalise_path(row.get("held_support", "")) for row in crossfit_manifest]
    crossfit_train = [
        [_normalise_path(path) for path in row.get("train_support", [])]
        if isinstance(row.get("train_support", []), list) else []
        for row in crossfit_manifest
    ]
    crossfit_is_strict = bool(
        len(crossfit_manifest) == len(support_paths) == 4
        and set(crossfit_held) == set(support_paths)
        and all(
            held not in train
            and set(train) == set(support_paths) - {held}
            and row.get("grounder_refit_without_held_support") is True
            for held, train, row in zip(crossfit_held, crossfit_train, crossfit_manifest)
        )
    )

    manifest_source_indices = [row.get("source_index") for row in manifest]
    source_indices_valid = all(
        isinstance(value, int) and not isinstance(value, bool)
        and 0 <= value < len(support_paths)
        for value in manifest_source_indices
    )
    manifest_counts = Counter(manifest_source_indices)
    test_references_in_manifest = _path_references(manifest, test_paths)
    test_references_in_library = _path_references(library, test_paths)
    manifest_absolute_paths = [
        {"location": location, "value": value}
        for location, value in _walk_strings(manifest)
        if Path(value).is_absolute()
    ]

    experiment_source = _experiment_source_audit(repo_root)
    runtime = _source_runtime_audit(
        repo_root,
        run_dir / "normality_rules_v3_relational.json",
        Path(test_paths[0]),
    )

    overall_bootstrap = _stratified_paired_bootstrap(
        labels,
        arrays["baseline"],
        {"relational": arrays["relational"], "fused": arrays["fused"]},
    )
    defect_types = np.asarray([row["defect_type"] for row in prediction_rows])
    cable_swap_mask = np.isin(defect_types, [*NORMAL_TYPES, "cable_swap"])
    if int((defect_types == "cable_swap").sum()) == 0:
        raise ValidationError("cable_swap rows are required for its registered CI")
    cable_swap_bootstrap = _stratified_paired_bootstrap(
        labels[cable_swap_mask],
        arrays["baseline"][cable_swap_mask],
        {"fused": arrays["fused"][cable_swap_mask]},
        metrics=("i_auroc",),
    )

    alpha = float(library.get("fusion", {}).get("alpha", 0.0))
    fusion_enabled = bool(library.get("fusion", {}).get("enabled", False))
    expected_decision = arrays["baseline"].copy()
    if fusion_enabled and library.get("rules"):
        active = arrays["relational"] > 0.0
        expected_decision[active] += alpha * arrays["relational"][active]
    decision_formula_max_error = float(
        np.max(np.abs(expected_decision - arrays["fused"]))
    )

    qwen_payload = library.get("qwen", {})
    qwen_payload_matches = bool(
        isinstance(qwen_payload, Mapping)
        and qwen_payload.get("model") == proposal.get("model")
        and qwen_payload.get("normality_summary") == proposal.get("normality_summary")
        and qwen_payload.get("entities") == proposal.get("entities")
        and qwen_payload.get("candidate_rules") == proposal.get("rules")
    )
    shot = int(saved.get("shot", -1))
    selected_rules = library.get("rules", [])
    if not isinstance(selected_rules, list):
        selected_rules = []
    model_hash_recorded = bool(recorded_model_hash)
    checks: Dict[str, bool] = {
        "required_artifacts_present": True,
        "schema_versions_are_v3_relational": (
            str(saved.get("schema_version")) == "3.0-relational"
            and str(library.get("schema_version")) == "3.0-relational"
        ),
        "binary_labels_with_both_classes": set(np.unique(labels).tolist()) == {0, 1},
        "all_prediction_scores_finite": all(
            np.isfinite(arrays[name]).all() for name in SCORE_COLUMNS
        ),
        "prediction_paths_unique": len(prediction_paths) == len(set(prediction_paths)),
        "split_test_paths_unique": len(test_paths) == len(set(test_paths)),
        "predictions_match_split_paths": predictions_match_split,
        "prediction_labels_match_split": labels_match_split,
        "prediction_types_match_split": types_match_split,
        "labels_match_good_vs_defect_paths": labels_match_paths,
        "support_test_disjoint": set(support_paths).isdisjoint(test_paths),
        "support_count_is_exactly_four": len(support_paths) == 4,
        "support_count_matches_saved_shot": len(support_paths) == shot,
        "support_paths_unique": len(support_paths) == len(set(support_paths)),
        "support_paths_are_train_good": all(
            "/train/good/" in path.replace("\\", "/").lower()
            or "/train/ok/" in path.replace("\\", "/").lower()
            for path in support_paths
        ),
        "all_support_and_test_files_exist": all(
            Path(path).is_file() for path in [*support_paths, *test_paths]
        ),
        "library_support_matches_split": support_paths == library_support,
        "library_declares_no_test_label_use": (
            library.get("test_labels_used_for_training_or_selection") is False
        ),
        "split_declares_no_test_label_use": (
            split.get("test_labels_used_for_training_or_selection") is False
        ),
        "metrics_declares_no_test_label_use": (
            saved.get("test_labels_used_for_training_or_selection") is False
        ),
        "rule_library_contains_no_test_paths": not test_references_in_library,
        "calibration_manifest_has_no_test_paths": not test_references_in_manifest,
        "calibration_manifest_has_no_absolute_paths": not manifest_absolute_paths,
        "calibration_manifest_source_indices_valid": source_indices_valid,
        "calibration_manifest_uses_all_and_only_four_supports": bool(
            source_indices_valid
            and len(support_paths) == 4
            and set(manifest_source_indices) == {0, 1, 2, 3}
        ),
        "calibration_manifest_has_normal_and_pseudo_rows": set(
            int(row.get("label", -1)) for row in manifest
        ) == {0, 1},
        "crossfit_manifest_has_four_strict_refit_folds": crossfit_is_strict,
        "library_records_strict_grounder_crossfit": (
            provenance.get("rule_cv_protocol")
            == "leave_one_support_out_refit_grounder"
        ),
        "calibration_source_code_consumes_support_paths": experiment_source[
            "relational_calibration_called_with_support_paths"
        ],
        "library_compiled_before_test_label_array": experiment_source[
            "library_compiled_before_test_label_array"
        ],
        "proposal_hash_matches_provenance": bool(proposal_expected_hash)
        and proposal_actual_hash == proposal_expected_hash,
        "qwen_proposal_payload_matches_library": qwen_payload_matches,
        "qwen_model_reference_matches_proposal_and_library": (
            qwen_model_from_proposal == qwen_model_from_library
        ),
        "qwen_model_directory_exists": qwen_model_path.is_dir(),
        "qwen_model_all_indexed_weight_shards_fully_hashed": bool(
            qwen_model_fingerprint["all_indexed_weight_shards_hashed"]
        ),
        "qwen_model_hash_matches_induction_provenance": bool(recorded_model_hash)
        and qwen_model_fingerprint["aggregate_sha256"] == recorded_model_hash,
        "checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_hash_matches_provenance": bool(checkpoint_expected_hash)
        and checkpoint_actual_hash == checkpoint_expected_hash,
        "support_hash_keyset_matches_split": set(expected_support_hashes)
        == set(support_paths),
        "all_support_hashes_match_provenance": bool(support_hash_rows)
        and all(row["matches"] for row in support_hash_rows),
        "library_declares_runtime_does_not_load_llm": (
            provenance.get("runtime_loads_llm") is False
        ),
        "library_records_complete_adapter_config": set(
            provenance.get("adapter_config", {})
        ) == {"inpformer_root", "encoder", "inp_num", "input_size", "crop_size", "map_size"},
        "runtime_source_has_no_llm_import_or_loader": runtime[
            "static_runtime_has_no_llm_dependency"
        ],
        "compiled_rules_execute_with_llm_imports_blocked": runtime[
            "compiled_rules_run_with_llm_blocked"
        ],
        "decision_scores_match_static_fusion_formula": (
            decision_formula_max_error <= metric_tolerance
        ),
        "saved_overall_and_per_defect_metrics_match_recomputation": (
            saved_errors["maximum_absolute_error"] <= metric_tolerance
        ),
        "saved_counts_match_predictions": (
            int(saved.get("test_count", -1)) == len(labels)
            and int(saved.get("normal_count", -1)) == int((labels == 0).sum())
            and int(saved.get("anomaly_count", -1)) == int((labels == 1).sum())
        ),
        "selected_rule_count_matches_library": int(
            saved.get("selected_rule_count", -1)
        ) == len(selected_rules),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    warnings = []
    if not model_hash_recorded:
        warnings.append(
            "The full Qwen model was hashed now, but the induction library did not "
            "persist an expected model-directory hash; historical byte identity cannot "
            "be proven from this run alone."
        )
    if str(saved.get("evaluation_status", "")) == "exploratory_development":
        warnings.append(
            "This is an exploratory, post-diagnosis cable run; bootstrap intervals are "
            "conditional on this fixed test set and do not establish cross-class gains."
        )
    selected_families = sorted({
        str(rule.get("family", "")) for rule in selected_rules if rule.get("family")
    })
    rejected_families = sorted({
        str(item.get("family", ""))
        for item in library.get("candidate_audit", [])
        if item.get("family") and not item.get("accepted", False)
    })
    warnings.append(
        "Empirically selected rule families are "
        + (", ".join(selected_families) if selected_families else "none")
        + "; proposed but rejected/disabled families include "
        + (", ".join(rejected_families) if rejected_families else "none")
        + ". This run does not establish efficacy for rejected families."
    )

    status = "needs_revision" if failed_checks else (
        "share_with_caveats" if warnings else "ready_within_reviewed_scope"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "validation_status": status,
        "valid": not failed_checks,
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "dataset": str(saved.get("dataset", "")),
        "class_name": str(saved.get("class_name", "")),
        "shot": shot,
        "population": {
            "test_count": int(len(labels)),
            "normal_count": int((labels == 0).sum()),
            "anomaly_count": int((labels == 1).sum()),
            "support_count": len(support_paths),
            "calibration_row_count": len(manifest),
            "calibration_rows_per_support": {
                str(key): int(value) for key, value in sorted(manifest_counts.items())
            },
        },
        "metrics_recomputed": scopes,
        "bootstrap": {
            "overall": overall_bootstrap,
            "cable_swap_vs_good": cable_swap_bootstrap,
            "cable_swap_fused_minus_baseline_point_delta_i_auroc": scopes[
                "cable_swap_vs_good"
            ]["fused_minus_baseline"]["delta_i_auroc"],
        },
        "saved_metric_validation": {
            **saved_errors,
            "tolerance": float(metric_tolerance),
        },
        "decision_formula_max_absolute_error": decision_formula_max_error,
        "split_and_calibration_evidence": {
            "support_paths": support_paths,
            "test_path_count": len(test_paths),
            "manifest_source_indices": sorted(set(manifest_source_indices)),
            "manifest_test_references": test_references_in_manifest,
            "library_test_references": test_references_in_library,
            "manifest_absolute_paths": manifest_absolute_paths,
            "experiment_source_audit": experiment_source,
            "crossfit_manifest": crossfit_manifest,
            "crossfit_is_strict": crossfit_is_strict,
        },
        "hash_evidence": {
            "qwen_proposal": {
                "canonical_sha256_expected": proposal_expected_hash,
                "canonical_sha256_actual": proposal_actual_hash,
                "artifact_file_sha256": proposal_artifact_hash,
                "raw_output_file_sha256": raw_output_hash,
                "support_montage_file_sha256": montage_hash,
            },
            "qwen_model": {
                **qwen_model_fingerprint,
                "recorded_expected_sha256": recorded_model_hash,
                "recorded_expected_sha256_available": model_hash_recorded,
                "recorded_expected_sha256_matches": (
                    qwen_model_fingerprint["aggregate_sha256"] == recorded_model_hash
                    if model_hash_recorded else None
                ),
            },
            "inpformer_checkpoint": {
                "path": str(checkpoint_path),
                "expected_sha256": checkpoint_expected_hash,
                "actual_sha256": checkpoint_actual_hash,
                "matches": checkpoint_actual_hash == checkpoint_expected_hash,
            },
            "supports": support_hash_rows,
        },
        "runtime_llm_independence": runtime,
        "checks": checks,
        "failed_checks": failed_checks,
        "warnings": warnings,
    }


def _csv_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    overall_bootstrap = report["bootstrap"]["overall"]
    cable_bootstrap = report["bootstrap"]["cable_swap_vs_good"]
    for scope_name, result in report["metrics_recomputed"].items():
        metrics = result["metrics"]
        overall_fused = overall_bootstrap["fused_minus_baseline"]
        overall_relation = overall_bootstrap["relational_minus_baseline"]
        cable_fused = cable_bootstrap["fused_minus_baseline"]
        row = {
            "validation_status": report["validation_status"],
            "valid": report["valid"],
            "scope": scope_name,
            "defect_type": result["defect_type"],
            "normal_count": result["normal_count"],
            "anomaly_count": result["anomaly_count"],
            "baseline_i_auroc": metrics["baseline"]["i_auroc"],
            "relational_i_auroc": metrics["relational"]["i_auroc"],
            "fused_i_auroc": metrics["fused"]["i_auroc"],
            "fused_delta_i_auroc": result["fused_minus_baseline"]["delta_i_auroc"],
            "relational_delta_i_auroc": result["relational_minus_baseline"][
                "delta_i_auroc"
            ],
            "baseline_i_aupr": metrics["baseline"]["i_aupr"],
            "relational_i_aupr": metrics["relational"]["i_aupr"],
            "fused_i_aupr": metrics["fused"]["i_aupr"],
            "fused_delta_i_aupr": result["fused_minus_baseline"]["delta_i_aupr"],
            "relational_delta_i_aupr": result["relational_minus_baseline"][
                "delta_i_aupr"
            ],
            "relation_trigger_rate_anomaly": result["relation_trigger_rate_anomaly"],
            "delta_i_auroc_ci95_low": "",
            "delta_i_auroc_ci95_high": "",
            "delta_i_aupr_ci95_low": "",
            "delta_i_aupr_ci95_high": "",
            "relational_delta_i_auroc_ci95_low": "",
            "relational_delta_i_auroc_ci95_high": "",
            "relational_delta_i_aupr_ci95_low": "",
            "relational_delta_i_aupr_ci95_high": "",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "checkpoint_sha256": "",
            "qwen_model_aggregate_sha256": "",
            "failed_checks": ";".join(report["failed_checks"]),
        }
        if scope_name == "overall":
            row["delta_i_auroc_ci95_low"], row["delta_i_auroc_ci95_high"] = (
                overall_fused["delta_i_auroc_ci95"]
            )
            row["delta_i_aupr_ci95_low"], row["delta_i_aupr_ci95_high"] = (
                overall_fused["delta_i_aupr_ci95"]
            )
            (
                row["relational_delta_i_auroc_ci95_low"],
                row["relational_delta_i_auroc_ci95_high"],
            ) = overall_relation["delta_i_auroc_ci95"]
            (
                row["relational_delta_i_aupr_ci95_low"],
                row["relational_delta_i_aupr_ci95_high"],
            ) = overall_relation["delta_i_aupr_ci95"]
            row["checkpoint_sha256"] = report["hash_evidence"][
                "inpformer_checkpoint"
            ]["actual_sha256"]
            row["qwen_model_aggregate_sha256"] = report["hash_evidence"][
                "qwen_model"
            ]["aggregate_sha256"]
        elif scope_name == "cable_swap_vs_good":
            row["delta_i_auroc_ci95_low"], row["delta_i_auroc_ci95_high"] = (
                cable_fused["delta_i_auroc_ci95"]
            )
        rows.append(row)
    return rows


def write_outputs(report: Mapping[str, Any], output_root: Path) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "validated_summary_v3_relational.json"
    csv_path = output_root / "validated_metrics_v3_relational.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = _csv_rows(report)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate a v3 relational FSAD experiment"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="results_v3/mvtec/cable/k4_qwen_identity",
    )
    parser.add_argument("--output-root", default="results_v3_validation")
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.metric_tolerance) or args.metric_tolerance < 0:
        raise SystemExit("--metric-tolerance must be finite and non-negative")
    report = validate_run(
        Path(args.run_dir), Path(args.output_root), metric_tolerance=args.metric_tolerance
    )
    json_path, csv_path = write_outputs(report, Path(args.output_root).resolve())
    receipt = {
        "validation_status": report["validation_status"],
        "valid": report["valid"],
        "failed_checks": report["failed_checks"],
        "summary_json": str(json_path),
        "metrics_csv": str(csv_path),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_invalid and not report["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
