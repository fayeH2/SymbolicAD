#!/usr/bin/env python3
"""Independently validate v2 neuro-symbolic FSAD experiment artifacts.

The validator deliberately does not import the experiment or rule-engine code.
Metrics are recomputed from ``predictions_v2.csv`` and data-leakage checks are
performed against the persisted split, rule library, and calibration manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


BOOTSTRAP_SEED = 20261013
DEFAULT_ITERATIONS = 1000
REQUIRED_FILES = (
    "metrics_v2.json",
    "predictions_v2.csv",
    "split.json",
    "normality_rules_v2.json",
    "calibration_manifest.json",
)
METRIC_GROUPS = {
    "baseline": "baseline",
    "rules": "rules_only",
    "decision": "baseline_plus_rules",
}
SCORE_COLUMNS = {
    "baseline": "baseline_score",
    "rules": "rule_score",
    "decision": "decision_score",
}


class ValidationError(ValueError):
    """Raised when an artifact cannot support the requested validation."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve(strict=False))


def _is_normal_train_path(value: str) -> bool:
    parts = [part.lower() for part in Path(value).parts]
    return any(
        parts[index] == "train" and parts[index + 1] in {"good", "ok"}
        for index in range(len(parts) - 1)
    )


def _safe_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValidationError("labels and scores must be same-length one-dimensional arrays")
    if len(labels) == 0:
        raise ValidationError("predictions_v2.csv is empty")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValidationError("evaluation requires both binary classes {0, 1}")
    if not np.isfinite(scores).all():
        raise ValidationError("prediction scores contain NaN or infinity")
    auroc = 0.5 if len(np.unique(scores)) < 2 else float(roc_auc_score(labels, scores))
    return {
        "i_auroc": auroc,
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    rules: np.ndarray,
    decision: np.ndarray,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Stratified paired bootstrap for rules/decision minus baseline deltas."""
    if iterations < 1:
        raise ValidationError("bootstrap iterations must be positive")
    labels = np.asarray(labels, dtype=int)
    scores = {
        "baseline": np.asarray(baseline, dtype=float),
        "rules": np.asarray(rules, dtype=float),
        "decision": np.asarray(decision, dtype=float),
    }
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    if len(normal) == 0 or len(anomaly) == 0:
        raise ValidationError("stratified bootstrap requires both normal and anomalous rows")

    rng = np.random.default_rng(seed)
    deltas = {
        "rules_minus_baseline": {
            "i_auroc": np.empty(iterations, dtype=float),
            "i_aupr": np.empty(iterations, dtype=float),
        },
        "decision_minus_baseline": {
            "i_auroc": np.empty(iterations, dtype=float),
            "i_aupr": np.empty(iterations, dtype=float),
        },
    }
    for iteration in range(iterations):
        chosen = np.concatenate((
            rng.choice(normal, size=len(normal), replace=True),
            rng.choice(anomaly, size=len(anomaly), replace=True),
        ))
        sampled_labels = labels[chosen]
        sampled = {
            name: _safe_metrics(sampled_labels, values[chosen])
            for name, values in scores.items()
        }
        for comparison, candidate in (
            ("rules_minus_baseline", "rules"),
            ("decision_minus_baseline", "decision"),
        ):
            for metric in ("i_auroc", "i_aupr"):
                deltas[comparison][metric][iteration] = (
                    sampled[candidate][metric] - sampled["baseline"][metric]
                )

    result: Dict[str, Any] = {
        "iterations": int(iterations),
        "seed": int(seed),
        "sampling": "stratified paired bootstrap with replacement",
    }
    for comparison, metric_values in deltas.items():
        result[comparison] = {
            f"delta_{metric}_ci95": [
                float(value) for value in np.quantile(values, [0.025, 0.975])
            ]
            for metric, values in metric_values.items()
        }
    return result


def _walk_strings(value: Any, location: str = "$") -> Iterable[Tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_strings(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{location}[{index}]")


def _manifest_rows(manifest: Any) -> List[Mapping[str, Any]]:
    if isinstance(manifest, list):
        rows = manifest
    elif isinstance(manifest, Mapping):
        rows = None
        for key in ("views", "samples", "calibration_samples", "manifest"):
            if isinstance(manifest.get(key), list):
                rows = manifest[key]
                break
        if rows is None:
            raise ValidationError("calibration manifest must contain a list of calibration rows")
    else:
        raise ValidationError("calibration manifest must be a list or object")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValidationError("calibration manifest rows must be non-empty JSON objects")
    return list(rows)


def _manifest_test_references(
    manifest: Any, test_paths_raw: Sequence[str], test_paths_resolved: Sequence[str]
) -> List[Dict[str, str]]:
    raw_targets = {str(value) for value in test_paths_raw}
    resolved_targets = set(test_paths_resolved)
    references = []
    for location, value in _walk_strings(manifest):
        raw_match = value in raw_targets or any(
            target and target in value for target in raw_targets
        )
        resolved_match = False
        # Synthetic identifiers such as normal:// and pseudo:// are not paths.
        if "://" not in value and (Path(value).is_absolute() or "/" in value or "\\" in value):
            resolved_match = _normalise_path(value) in resolved_targets
        if raw_match or resolved_match:
            references.append({"location": location, "value": value})
    return references


def _read_predictions(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", *SCORE_COLUMNS.values()}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValidationError(f"predictions_v2.csv missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValidationError("predictions_v2.csv contains no rows")
    try:
        labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
        arrays = {
            name: np.asarray([float(row[column]) for row in rows], dtype=float)
            for name, column in SCORE_COLUMNS.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid numeric value in predictions_v2.csv: {exc}") from exc
    arrays["labels"] = labels
    return rows, arrays


def _metric_error(
    saved: Mapping[str, Any], recomputed: Mapping[str, Mapping[str, float]]
) -> Tuple[float, Dict[str, float]]:
    errors: Dict[str, float] = {}
    for result_name, saved_name in METRIC_GROUPS.items():
        saved_group = saved.get(saved_name)
        if not isinstance(saved_group, Mapping):
            raise ValidationError(f"metrics_v2.json missing metric group: {saved_name}")
        for metric in ("i_auroc", "i_aupr"):
            try:
                stored = float(saved_group[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(f"invalid saved metric {saved_name}.{metric}") from exc
            errors[f"{saved_name}.{metric}"] = abs(recomputed[result_name][metric] - stored)

    decision_delta = {
        metric: recomputed["decision"][metric] - recomputed["baseline"][metric]
        for metric in ("i_auroc", "i_aupr")
    }
    for metric, saved_key in (("i_auroc", "delta_i_auroc"), ("i_aupr", "delta_i_aupr")):
        try:
            stored = float(saved[saved_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"invalid saved metric {saved_key}") from exc
        errors[saved_key] = abs(decision_delta[metric] - stored)
    return max(errors.values(), default=0.0), errors


def _empty_failure(run_dir: Path, message: str, missing_files: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "dataset": "",
        "class_name": "",
        "shot": None,
        "valid": False,
        "errors": [message],
        "missing_files": list(missing_files),
        "checks": {"required_files_present": not missing_files},
    }


def validate_run(
    run_dir: Any,
    iterations: int = DEFAULT_ITERATIONS,
    metric_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve(strict=False)
    missing_files = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing_files:
        return _empty_failure(
            run_dir,
            "required v2 artifacts are missing",
            missing_files=missing_files,
        )

    try:
        saved = _read_json(run_dir / "metrics_v2.json")
        split = _read_json(run_dir / "split.json")
        library = _read_json(run_dir / "normality_rules_v2.json")
        manifest = _read_json(run_dir / "calibration_manifest.json")
        manifest_rows = _manifest_rows(manifest)
        prediction_rows, arrays = _read_predictions(run_dir / "predictions_v2.csv")

        if not isinstance(saved, Mapping) or not isinstance(split, Mapping) \
                or not isinstance(library, Mapping):
            raise ValidationError("metrics, split, and rule library must be JSON objects")
        support_raw = split.get("support")
        split_test = split.get("test")
        if not isinstance(support_raw, list) or not isinstance(split_test, list):
            raise ValidationError("split.json must contain support and test lists")
        if not all(isinstance(item, Mapping) and "path" in item and "label" in item
                   for item in split_test):
            raise ValidationError("each split test row must contain path and label")

        labels = arrays["labels"]
        recomputed = {
            name: _safe_metrics(labels, arrays[name])
            for name in ("baseline", "rules", "decision")
        }
        maximum_error, metric_errors = _metric_error(saved, recomputed)

        prediction_paths_raw = [str(row["path"]) for row in prediction_rows]
        prediction_paths = [_normalise_path(path) for path in prediction_paths_raw]
        split_test_paths_raw = [str(item["path"]) for item in split_test]
        split_test_paths = [_normalise_path(path) for path in split_test_paths_raw]
        support_paths = [_normalise_path(path) for path in support_raw]
        library_support_raw = library.get("support_paths", [])
        if not isinstance(library_support_raw, list):
            raise ValidationError("normality_rules_v2.json support_paths must be a list")
        library_support = [_normalise_path(path) for path in library_support_raw]

        split_label_by_path: Dict[str, int] = {}
        split_labels_parseable = True
        for item, path in zip(split_test, split_test_paths):
            try:
                split_label_by_path[path] = int(item["label"])
            except (TypeError, ValueError):
                split_labels_parseable = False
        prediction_labels_match = bool(
            split_labels_parseable
            and len(prediction_paths) == len(set(prediction_paths))
            and all(
                path in split_label_by_path and int(row["label"]) == split_label_by_path[path]
                for row, path in zip(prediction_rows, prediction_paths)
            )
        )

        manifest_references = _manifest_test_references(
            manifest, split_test_paths_raw, split_test_paths
        )
        source_indices = [row.get("source_index") for row in manifest_rows]
        source_indices_valid = all(
            isinstance(index, int) and not isinstance(index, bool)
            and 0 <= index < len(support_paths)
            for index in source_indices
        )
        source_indices_cover_support = bool(
            source_indices_valid and set(source_indices) == set(range(len(support_paths)))
        )
        identifiers = [row.get("identifier") for row in manifest_rows]
        identifiers_valid = all(isinstance(value, str) and value for value in identifiers)
        identifiers_unique = identifiers_valid and len(identifiers) == len(set(identifiers))
        manifest_labels = [row.get("label") for row in manifest_rows]
        manifest_labels_binary = all(value in (0, 1, False, True) for value in manifest_labels)
        manifest_has_both_classes = manifest_labels_binary and set(map(int, manifest_labels)) == {0, 1}

        fusion = library.get("fusion", {})
        if not isinstance(fusion, Mapping):
            fusion = {}
        training_device = str(fusion.get("training_device", ""))
        fusion_device_cuda = bool(re.fullmatch(r"cuda(?::\d+)?", training_device.lower()))

        shot = int(saved.get("shot", -1))
        dataset = str(saved.get("dataset", ""))
        class_name = str(saved.get("class_name", ""))
        saved_counts_match = all((
            int(saved.get("test_count", -1)) == len(labels),
            int(saved.get("normal_count", -1)) == int((labels == 0).sum()),
            int(saved.get("anomaly_count", -1)) == int((labels == 1).sum()),
        ))
        metadata_consistent = bool(
            dataset
            and class_name
            and str(library.get("dataset", "")) == dataset
            and str(library.get("class_name", "")) == class_name
        )
        selected_rule_count_matches = int(saved.get("selected_rule_count", -1)) == len(
            library.get("rules", []) if isinstance(library.get("rules"), list) else []
        )

        checks = {
            "required_files_present": True,
            "schema_versions_v2": (
                str(saved.get("schema_version")) == "2.0"
                and str(library.get("schema_version")) == "2.0"
            ),
            "prediction_scores_finite": all(
                np.isfinite(arrays[name]).all() for name in ("baseline", "rules", "decision")
            ),
            "labels_are_binary_with_both_classes": set(np.unique(labels).tolist()) == {0, 1},
            "prediction_paths_unique": len(prediction_paths) == len(set(prediction_paths)),
            "split_test_paths_unique": len(split_test_paths) == len(set(split_test_paths)),
            "predictions_match_split_test_paths": (
                len(prediction_paths) == len(split_test_paths)
                and set(prediction_paths) == set(split_test_paths)
            ),
            "prediction_labels_match_split": prediction_labels_match,
            "support_test_disjoint": set(support_paths).isdisjoint(split_test_paths),
            "support_is_normal_train": bool(support_paths) and all(
                _is_normal_train_path(path) for path in support_paths
            ),
            "support_paths_exist": bool(support_paths) and all(
                Path(path).is_file() for path in support_paths
            ),
            "test_paths_exist": bool(split_test_paths) and all(
                Path(path).is_file() for path in split_test_paths
            ),
            "support_matches_shot": len(support_paths) == shot,
            "library_support_matches_split": (
                len(library_support) == len(support_paths)
                and set(library_support) == set(support_paths)
            ),
            "support_only_calibration_declared": library.get("support_only_calibration") is True,
            "split_declares_no_test_label_use": (
                split.get("test_labels_used_for_training_or_selection") is False
            ),
            "calibration_manifest_no_test_paths": not manifest_references,
            "calibration_source_indices_valid": source_indices_valid,
            "calibration_source_indices_cover_support": source_indices_cover_support,
            "calibration_identifiers_unique": identifiers_unique,
            "calibration_labels_binary_with_both_classes": manifest_has_both_classes,
            "fusion_training_device_is_cuda": fusion_device_cuda,
            "saved_counts_match_predictions": saved_counts_match,
            "saved_metadata_matches_rule_library": metadata_consistent,
            "selected_rule_count_matches_library": selected_rule_count_matches,
            "saved_metrics_match_recomputation": maximum_error <= metric_tolerance,
        }
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        bootstrap = _paired_bootstrap(
            labels,
            arrays["baseline"],
            arrays["rules"],
            arrays["decision"],
            iterations=iterations,
            seed=BOOTSTRAP_SEED,
        )
        decision_delta = {
            metric: recomputed["decision"][metric] - recomputed["baseline"][metric]
            for metric in ("i_auroc", "i_aupr")
        }
        rules_delta = {
            metric: recomputed["rules"][metric] - recomputed["baseline"][metric]
            for metric in ("i_auroc", "i_aupr")
        }
        return {
            "run_dir": str(run_dir),
            "dataset": dataset,
            "class_name": class_name,
            "shot": shot,
            "test_count": int(len(labels)),
            "normal_count": int((labels == 0).sum()),
            "anomaly_count": int((labels == 1).sum()),
            "support_count": len(support_paths),
            "calibration_count": len(manifest_rows),
            "fusion_enabled": bool(fusion.get("enabled", False)),
            "fusion_training_device": training_device,
            "metrics_recomputed": recomputed,
            "rules_minus_baseline": {
                "delta_i_auroc": rules_delta["i_auroc"],
                "delta_i_aupr": rules_delta["i_aupr"],
            },
            "decision_minus_baseline": {
                "delta_i_auroc": decision_delta["i_auroc"],
                "delta_i_aupr": decision_delta["i_aupr"],
            },
            "bootstrap": bootstrap,
            "saved_metric_max_abs_error": float(maximum_error),
            "saved_metric_abs_errors": metric_errors,
            "metric_tolerance": float(metric_tolerance),
            "manifest_test_references": manifest_references,
            "checks": checks,
            "failed_checks": failed_checks,
            "errors": [],
            "valid": not failed_checks,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _empty_failure(run_dir, f"{type(exc).__name__}: {exc}")


def _macro_summary(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [row for row in results if isinstance(row.get("metrics_recomputed"), Mapping)]
    if not eligible:
        return {"run_count_with_recomputed_metrics": 0}
    output: Dict[str, Any] = {"run_count_with_recomputed_metrics": len(eligible)}
    for metric in ("i_auroc", "i_aupr"):
        for group in ("baseline", "rules", "decision"):
            output[f"macro_{group}_{metric}"] = float(np.mean([
                row["metrics_recomputed"][group][metric] for row in eligible
            ]))
        output[f"macro_decision_minus_baseline_delta_{metric}"] = (
            output[f"macro_decision_{metric}"] - output[f"macro_baseline_{metric}"]
        )
        output[f"macro_rules_minus_baseline_delta_{metric}"] = (
            output[f"macro_rules_{metric}"] - output[f"macro_baseline_{metric}"]
        )
    return output


def _csv_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics_recomputed", {})
    bootstrap = result.get("bootstrap", {})
    decision_bootstrap = bootstrap.get("decision_minus_baseline", {})
    rules_bootstrap = bootstrap.get("rules_minus_baseline", {})

    def metric(group: str, name: str) -> Any:
        return metrics.get(group, {}).get(name, "")

    def bound(payload: Mapping[str, Any], key: str, index: int) -> Any:
        values = payload.get(key, [])
        return values[index] if isinstance(values, list) and len(values) == 2 else ""

    return {
        "run_dir": result.get("run_dir", ""),
        "dataset": result.get("dataset", ""),
        "class_name": result.get("class_name", ""),
        "shot": result.get("shot", ""),
        "test_count": result.get("test_count", ""),
        "normal_count": result.get("normal_count", ""),
        "anomaly_count": result.get("anomaly_count", ""),
        "baseline_i_auroc": metric("baseline", "i_auroc"),
        "rules_i_auroc": metric("rules", "i_auroc"),
        "decision_i_auroc": metric("decision", "i_auroc"),
        "decision_delta_i_auroc": result.get("decision_minus_baseline", {}).get("delta_i_auroc", ""),
        "decision_delta_i_auroc_ci95_low": bound(
            decision_bootstrap, "delta_i_auroc_ci95", 0
        ),
        "decision_delta_i_auroc_ci95_high": bound(
            decision_bootstrap, "delta_i_auroc_ci95", 1
        ),
        "rules_delta_i_auroc": result.get("rules_minus_baseline", {}).get("delta_i_auroc", ""),
        "rules_delta_i_auroc_ci95_low": bound(rules_bootstrap, "delta_i_auroc_ci95", 0),
        "rules_delta_i_auroc_ci95_high": bound(rules_bootstrap, "delta_i_auroc_ci95", 1),
        "baseline_i_aupr": metric("baseline", "i_aupr"),
        "rules_i_aupr": metric("rules", "i_aupr"),
        "decision_i_aupr": metric("decision", "i_aupr"),
        "decision_delta_i_aupr": result.get("decision_minus_baseline", {}).get("delta_i_aupr", ""),
        "decision_delta_i_aupr_ci95_low": bound(
            decision_bootstrap, "delta_i_aupr_ci95", 0
        ),
        "decision_delta_i_aupr_ci95_high": bound(
            decision_bootstrap, "delta_i_aupr_ci95", 1
        ),
        "rules_delta_i_aupr": result.get("rules_minus_baseline", {}).get("delta_i_aupr", ""),
        "rules_delta_i_aupr_ci95_low": bound(rules_bootstrap, "delta_i_aupr_ci95", 0),
        "rules_delta_i_aupr_ci95_high": bound(rules_bootstrap, "delta_i_aupr_ci95", 1),
        "fusion_training_device": result.get("fusion_training_device", ""),
        "support_test_disjoint": result.get("checks", {}).get("support_test_disjoint", ""),
        "support_is_normal_train": result.get("checks", {}).get("support_is_normal_train", ""),
        "calibration_manifest_no_test_paths": result.get("checks", {}).get(
            "calibration_manifest_no_test_paths", ""
        ),
        "fusion_training_device_is_cuda": result.get("checks", {}).get(
            "fusion_training_device_is_cuda", ""
        ),
        "saved_metric_max_abs_error": result.get("saved_metric_max_abs_error", ""),
        "valid": result.get("valid", False),
        "failed_checks": ";".join(result.get("failed_checks", [])),
        "errors": ";".join(result.get("errors", [])),
    }


def write_summary(
    results: Sequence[Mapping[str, Any]], output_root: Any
) -> Tuple[Path, Path, Dict[str, Any]]:
    output_root = Path(output_root).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    all_valid = bool(results) and all(bool(row.get("valid")) for row in results)
    payload = {
        "schema_version": "2.0-validation",
        "validation_status": (
            "ready_within_reviewed_scope" if all_valid else "needs_revision"
        ),
        "run_count": len(results),
        "valid_run_count": sum(bool(row.get("valid")) for row in results),
        "bootstrap": {
            "iterations": (
                results[0].get("bootstrap", {}).get("iterations", DEFAULT_ITERATIONS)
                if results else DEFAULT_ITERATIONS
            ),
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "method": "stratified paired percentile bootstrap",
        },
        "macro": _macro_summary(results),
        "runs": list(results),
    }
    json_path = output_root / "validated_summary_v2.json"
    csv_path = output_root / "validated_summary_v2.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = [_csv_row(row) for row in results]
    fields = list(csv_rows[0]) if csv_rows else list(_csv_row({}))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return json_path, csv_path, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate one or more NeuroSymbolic FSAD v2 runs"
    )
    parser.add_argument("run_dirs", nargs="+", help="Directories containing the five v2 artifacts")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    parser.add_argument("--output-root", default="results_v2_validation")
    parser.add_argument(
        "--fail-on-invalid", action="store_true",
        help="Exit with status 1 after writing outputs when any run is invalid",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    if not math.isfinite(args.metric_tolerance) or args.metric_tolerance < 0:
        raise SystemExit("--metric-tolerance must be a finite non-negative number")
    results = [
        validate_run(path, iterations=args.iterations, metric_tolerance=args.metric_tolerance)
        for path in args.run_dirs
    ]
    json_path, csv_path, payload = write_summary(results, args.output_root)
    receipt = {
        "validation_status": payload["validation_status"],
        "run_count": payload["run_count"],
        "valid_run_count": payload["valid_run_count"],
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if args.fail_on_invalid and payload["validation_status"] != "ready_within_reviewed_scope":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
