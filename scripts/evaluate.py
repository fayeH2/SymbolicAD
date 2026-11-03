#!/usr/bin/env python
"""Recompute metrics from saved ``results/predictions/{category}.npz`` files."""


from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from symbolicAD.config import load_config
from symbolicAD.evaluation.evaluator import CategoryEvaluator


def _evaluate_file(path: Path, config) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        evaluator = CategoryEvaluator(config)
        evaluator.labels = payload["labels"].tolist()
        evaluator.groups = [str(item) for item in payload["groups"].tolist()]
        evaluator.masks = [item for item in payload["masks"]]
        for variant in ("baseline", "logic", "final"):
            evaluator.image_scores[variant] = payload[f"{variant}_scores"].tolist()
            evaluator.maps[variant] = [item for item in payload[f"{variant}_maps"]]
        metrics = evaluator.compute()
    metrics["category"] = path.stem
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved SymbolicAD predictions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.results_dir:
        config.paths.results_dir = args.results_dir
    predictions_dir = (
        Path(args.predictions_dir)
        if args.predictions_dir
        else config.results_dir / "predictions"
    )
    files = sorted(predictions_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no prediction files under {predictions_dir}")

    all_metrics = {}
    rows = []
    for path in files:
        metrics = _evaluate_file(path, config)
        all_metrics[path.stem] = metrics
        for group, group_payload in metrics["groups"].items():
            for variant in ("baseline", "logic", "final"):
                item = group_payload[variant]
                rows.append(
                    {
                        "category": path.stem,
                        "group": group,
                        "variant": variant,
                        "image_auroc": item["image"].get("auroc"),
                        "image_ap": item["image"].get("ap"),
                        "image_f1max": item["image"].get("f1max"),
                        "pixel_auroc": item["pixel"].get("auroc"),
                        "pixel_ap": item["pixel"].get("ap"),
                        "pixel_f1max": item["pixel"].get("f1max"),
                        "aupro": item.get("pro", {}).get("aupro"),
                        "n_samples": group_payload.get("n_samples"),
                    }
                )

    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics_summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0]) if rows else []
    with (results_dir / "metrics_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {results_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
