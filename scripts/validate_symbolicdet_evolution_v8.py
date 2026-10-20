#!/usr/bin/env python3
"""Independent metric, leakage, trace, and bootstrap audit for v8."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from validate_executable_logic_v7 import (
    METRICS,
    _bootstrap,
    _hist_metric,
    _image_histograms,
    _metric,
)


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_nodes(nodes: list[Mapping[str, Any]], tolerance: float = 1e-10):
    outputs: dict[str, float] = {}
    errors = []
    atom_counts: Counter[str] = Counter()
    for node in nodes:
        node_id = str(node["node_id"])
        op = str(node["op"]).upper()
        inputs = [outputs[str(value)] for value in node.get("input_node_ids", [])]
        observed = float(node["output"])
        if op == "ATOM":
            expected = observed
            atom_counts[str(node["atom"])] += 1
        elif op == "NOT":
            expected = 1.0 - inputs[0]
        elif op == "AND":
            expected = float(np.prod(inputs))
        elif op == "OR":
            expected = float(1.0 - np.prod([1.0 - value for value in inputs]))
        elif op == "IMPLIES":
            expected = float(1.0 - inputs[0] * (1.0 - inputs[1]))
        elif op == "XOR":
            expected = float(inputs[0] + inputs[1] - 2.0 * inputs[0] * inputs[1])
        else:
            errors.append({"node_id": node_id, "error": f"unknown op {op}"})
            expected = observed
        if abs(expected - observed) > tolerance:
            errors.append({
                "node_id": node_id,
                "expected": expected,
                "observed": observed,
            })
        outputs[node_id] = observed
    return outputs, atom_counts, errors


def validate(run_dir: Path, repeats: int, bins: int, seed: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    metrics = json.loads((run_dir / "metrics_v8_symbolicdet.json").read_text())
    library = json.loads(
        (run_dir / "normality_rules_v8_symbolicdet_evolved.json").read_text()
    )
    search = json.loads((run_dir / "evolution_search_audit_v8.json").read_text())
    rows = _rows(run_dir / "predictions_v8_symbolicdet.csv")
    with np.load(run_dir / "pixel_predictions_v8.npz", allow_pickle=False) as payload:
        paths = payload["paths"].astype(str)
        labels = payload["labels"].astype(np.uint8)
        masks = payload["masks"].astype(bool)
        baseline_maps = payload["baseline_maps"].astype(np.float32)
        evolved_maps = payload["evolved_maps"].astype(np.float32)
    row_paths = np.asarray([str(Path(row["path"]).resolve()) for row in rows])
    if not np.array_equal(paths, row_paths):
        raise ValueError("CSV and NPZ paths differ")
    csv_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if not np.array_equal(labels, csv_labels):
        raise ValueError("CSV and NPZ labels differ")
    baseline_scores = np.asarray([float(row["baseline_score"]) for row in rows])
    evolved_scores = np.asarray([float(row["evolved_score"]) for row in rows])
    baseline_i = _metric(labels, baseline_scores)
    evolved_i = _metric(labels, evolved_scores)
    baseline_p = _metric(masks.reshape(-1), baseline_maps.reshape(-1))
    evolved_p = _metric(masks.reshape(-1), evolved_maps.reshape(-1))
    recomputed = {
        "baseline": dict(zip(METRICS, (*baseline_i, *baseline_p))),
        "evolved_neuro_symbolic": dict(zip(METRICS, (*evolved_i, *evolved_p))),
    }
    maximum_error = max(
        abs(recomputed[method][key] - float(metrics[method][key]))
        for method in recomputed for key in METRICS
    )
    baseline_hist = _image_histograms(baseline_maps, masks, bins)
    evolved_hist = _image_histograms(evolved_maps, masks, bins)
    full_hist = {
        "baseline": dict(zip(("p_auroc", "p_aupr"), _hist_metric(
            baseline_hist[0].sum(axis=0), baseline_hist[1].sum(axis=0)
        ))),
        "evolved_neuro_symbolic": dict(zip(("p_auroc", "p_aupr"), _hist_metric(
            evolved_hist[0].sum(axis=0), evolved_hist[1].sum(axis=0)
        ))),
    }
    histogram_max_error = max(
        abs(full_hist[method][key] - recomputed[method][key])
        for method in full_hist for key in ("p_auroc", "p_aupr")
    )
    support = {str(Path(path).resolve()) for path in library["support"]}
    overlap = sorted(support.intersection(set(paths.tolist())))

    active = list(library["accepted_rule_ids"])
    trace_files = sorted((run_dir / "logic_traces").glob("*.json"))
    trace_errors = []
    total_nodes = 0
    for trace_path in trace_files:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        symbolic_nodes = trace["symbolic_logic_nodes"]
        final_nodes = trace["final_fusion_logic_nodes"]
        symbolic_outputs, counts, errors = _audit_nodes(symbolic_nodes)
        final_outputs, _, final_errors = _audit_nodes(final_nodes)
        total_nodes += len(symbolic_nodes) + len(final_nodes)
        if any(counts[name] != 1 for name in active):
            errors.append({"active_atom_counts": dict(counts), "expected": active})
        if abs(symbolic_outputs[str(symbolic_nodes[-1]["node_id"])] - float(trace["symbolic_anomaly"])) > 1e-10:
            errors.append({"error": "symbolic root mismatch"})
        if abs(final_outputs[str(final_nodes[-1]["node_id"])] - float(trace["final_anomaly"])) > 1e-10:
            final_errors.append({"error": "final root mismatch"})
        if errors or final_errors:
            trace_errors.append({
                "path": trace["path"], "symbolic": errors, "final": final_errors
            })

    protocol = search.get("protocol", {})
    result = {
        "schema_version": "8.0-independent-symbolicdet-evolution-validation",
        "run_dir": str(run_dir),
        "query_count": len(rows),
        "exact_metric_recomputation": recomputed,
        "maximum_exact_metric_error": maximum_error,
        "exact_metrics_match": maximum_error <= 1e-12,
        "support_query_path_overlap": overlap,
        "support_query_disjoint": not overlap,
        "search_protocol_claims": {
            "test_images_seen_during_search": protocol.get("test_images_seen_during_search"),
            "test_labels_seen_during_search": protocol.get("test_labels_seen_during_search"),
            "query_or_test_data_used_in_library": library.get("rule_discovery", {}).get(
                "query_or_test_data_used"
            ),
        },
        "llm_feedback": {
            "rounds": len(search.get("llm_feedback", [])),
            "valid_suggestions": sum(
                int(row.get("valid_suggestion_count", 0))
                for row in search.get("llm_feedback", [])
            ),
            "all_rounds_exclude_query_test_information": all(
                row.get("query_or_test_information_supplied") is False
                for row in search.get("llm_feedback", [])
            ),
        },
        "logic_trace_audit": {
            "trace_count": len(trace_files),
            "total_node_count": total_nodes,
            "active_atoms": active,
            "error_count": len(trace_errors),
            "errors": trace_errors[:20],
        },
        "bootstrap": {
            "unit": "stratified query-image block",
            "repeats": repeats,
            "seed": seed,
            "pixel_histogram_bins": bins,
            "full_sample_histogram_max_error": histogram_max_error,
            "delta_evolved_minus_baseline": _bootstrap(
                labels, baseline_scores, evolved_scores,
                baseline_hist, evolved_hist, repeats, seed,
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--pixel-histogram-bins", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=668)
    args = parser.parse_args()
    result = validate(args.run_dir, args.repeats, args.pixel_histogram_bins, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
