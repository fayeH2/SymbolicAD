#!/usr/bin/env python3
"""Export one v7 executable-logic run without overwriting existing methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = ("i_auroc", "i_aupr", "p_auroc", "p_aupr")


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        raise FileExistsError(f"Refusing to retarget existing link: {link}")
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing output: {link}")
    link.symlink_to(target)


def export(run_dir: Path, destination: Path, seed: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    destination = destination.expanduser().resolve()
    metrics_path = run_dir / "metrics_v7_executable_logic.json"
    library_path = run_dir / "normality_rules_v7_executable_logic.json"
    metrics = _read(metrics_path)
    library = _read(library_path)
    class_name = str(metrics["class_name"])
    baseline = metrics["baseline"]
    fused = metrics["neuro_symbolic"]
    delta = metrics["delta_neuro_symbolic_minus_baseline"]

    csv_dir = destination / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    compact_path = csv_dir / f"Seed_{seed}-results.csv"
    with compact_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ClassName", "i_roc", "i_ap", "p_roc", "p_ap"])
        writer.writerow([
            class_name,
            100.0 * float(fused["i_auroc"]),
            100.0 * float(fused["i_aupr"]),
            100.0 * float(fused["p_auroc"]),
            100.0 * float(fused["p_aupr"]),
        ])
    detailed_path = csv_dir / f"Seed_{seed}-results-detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "class_name", "baseline_shot", "rule_support_count", "metric",
            "baseline", "neuro_symbolic", "delta", "accepted_rules",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in METRICS:
            writer.writerow({
                "class_name": class_name,
                "baseline_shot": metrics["baseline_shot"],
                "rule_support_count": metrics["rule_support_count"],
                "metric": key,
                "baseline": baseline[key],
                "neuro_symbolic": fused[key],
                "delta": delta[key],
                "accepted_rules": "|".join(metrics["accepted_rule_ids"]),
            })

    for source in sorted((run_dir / "visualizations").glob("*.jpg")):
        _link(destination / "imgs" / f"{class_name}-{source.name}", source)
    for source in sorted((run_dir / "logic_traces").glob("*.json")):
        _link(destination / "logic_traces" / f"{class_name}-{source.name}", source)

    checkpoint_dir = destination / "checkpoint"
    _link(
        checkpoint_dir / f"RULE-Seed_{seed}-{class_name}.json", library_path
    )
    candidate_bank = Path(str(metrics["candidate_bank"]))
    _link(
        checkpoint_dir / f"QWEN-CANDIDATES-Seed_{seed}-{class_name}.json",
        candidate_bank,
    )
    validation_path = run_dir / "validation_v7.json"
    if validation_path.is_file():
        _link(
            checkpoint_dir / f"VALIDATION-Seed_{seed}-{class_name}.json",
            validation_path,
        )
    checkpoint = library.get("baseline", {}).get("checkpoint")
    if checkpoint and Path(str(checkpoint)).is_file():
        _link(
            checkpoint_dir / f"CLS-Seed_{seed}-{class_name}-check_point.pth",
            Path(str(checkpoint)),
        )

    manifest = {
        "schema_version": "7.0-h2vlr-executable-logic-export",
        "method": "Qwen-static-executable-neuro-symbolic-FSAD",
        "source_run": str(run_dir),
        "destination": str(destination),
        "dataset": metrics["dataset"],
        "class_name": class_name,
        "baseline_shot": metrics["baseline_shot"],
        "rule_support_count": metrics["rule_support_count"],
        "support_seed": metrics["support_seed"],
        "accepted_rules": metrics["accepted_rule_ids"],
        "logic_formula": metrics["logic_formula"],
        "metrics_fraction": {"baseline": baseline, "neuro_symbolic": fused, "delta": delta},
        "compact_csv_units": "percent",
        "visualizations": len(list((destination / "imgs").glob("*.jpg"))),
        "query_logic_traces": len(list((destination / "logic_traces").glob("*.json"))),
        "runtime_requires_llm": False,
        "test_labels_used_for_rule_selection_or_calibration": False,
        "validation": str(validation_path) if validation_path.is_file() else None,
    }
    manifest_path = checkpoint_dir / f"Seed_{seed}-{class_name}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": str(manifest_path), "compact_csv": str(compact_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.destination, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
