#!/usr/bin/env python3
"""Export a v8 evolutionary rule run into the requested H2VLR layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("i_auroc", "i_aupr", "p_auroc", "p_aupr")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing artifact: {link}")
    link.symlink_to(target)


def export(run_dir: Path, destination: Path, seed: int):
    run_dir = run_dir.expanduser().resolve()
    destination = destination.expanduser().resolve()
    metrics_path = run_dir / "metrics_v8_symbolicdet.json"
    library_path = run_dir / "normality_rules_v8_symbolicdet_evolved.json"
    search_path = run_dir / "evolution_search_audit_v8.json"
    validation_path = run_dir / "validation_v8.json"
    metrics, library, search = _load(metrics_path), _load(library_path), _load(search_path)
    class_name = str(metrics["class_name"])
    baseline = metrics["baseline"]
    evolved = metrics["evolved_neuro_symbolic"]
    delta = metrics["delta_evolved_minus_baseline"]

    csv_dir = destination / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    compact = csv_dir / f"Seed_{seed}-results.csv"
    with compact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ClassName", "i_roc", "i_ap", "p_roc", "p_ap"])
        writer.writerow([
            class_name,
            100.0 * evolved["i_auroc"],
            100.0 * evolved["i_aupr"],
            100.0 * evolved["p_auroc"],
            100.0 * evolved["p_aupr"],
        ])
    detailed = csv_dir / f"Seed_{seed}-results-detailed.csv"
    with detailed.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "class_name", "baseline_shot", "rule_support_count", "metric",
            "baseline", "evolved_neuro_symbolic", "delta", "evolved_rules",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in METRICS:
            writer.writerow({
                "class_name": class_name,
                "baseline_shot": metrics["baseline_shot"],
                "rule_support_count": metrics["rule_support_count"],
                "metric": name,
                "baseline": baseline[name],
                "evolved_neuro_symbolic": evolved[name],
                "delta": delta[name],
                "evolved_rules": "|".join(metrics["evolved_active_atoms"]),
            })

    for source in sorted((run_dir / "visualizations").glob("*.jpg")):
        _link(destination / "imgs" / f"{class_name}-{source.name}", source)
    for source in sorted((run_dir / "logic_traces").glob("*.json")):
        _link(destination / "logic_traces" / f"{class_name}-{source.name}", source)
    checkpoint = destination / "checkpoint"
    _link(checkpoint / f"RULE-Seed_{seed}-{class_name}.json", library_path)
    _link(checkpoint / f"EVOLUTION-Seed_{seed}-{class_name}.json", search_path)
    _link(
        checkpoint / f"PREDICTIONS-Seed_{seed}-{class_name}.csv",
        run_dir / "predictions_v8_symbolicdet.csv",
    )
    if validation_path.is_file():
        _link(checkpoint / f"VALIDATION-Seed_{seed}-{class_name}.json", validation_path)
    source_library = Path(search["base_rule_library"])
    _link(checkpoint / f"SOURCE-RULES-Seed_{seed}-{class_name}.json", source_library)
    baseline_checkpoint = library.get("baseline", {}).get("checkpoint")
    if baseline_checkpoint and Path(baseline_checkpoint).is_file():
        _link(
            checkpoint / f"CLS-Seed_{seed}-{class_name}-check_point.pth",
            Path(baseline_checkpoint),
        )

    manifest = {
        "schema_version": "8.0-h2vlr-symbolicdet-evolution-export",
        "method": "Qwen-guided-evolutionary-static-neuro-symbolic-FSAD",
        "source_run": str(run_dir),
        "destination": str(destination),
        "dataset": metrics["dataset"],
        "class_name": class_name,
        "baseline_shot": metrics["baseline_shot"],
        "rule_support_count": metrics["rule_support_count"],
        "evolved_normality_ast": metrics["evolved_normality_ast"],
        "evolved_active_atoms": metrics["evolved_active_atoms"],
        "metrics_fraction": {
            "baseline": baseline,
            "evolved_neuro_symbolic": evolved,
            "delta": delta,
        },
        "compact_csv_units": "percent",
        "visualizations": len(list((destination / "imgs").glob("*.jpg"))),
        "query_logic_traces": len(list((destination / "logic_traces").glob("*.json"))),
        "runtime_requires_llm": False,
        "query_or_test_used_for_search": False,
        "validation": str(validation_path) if validation_path.is_file() else None,
    }
    manifest_path = checkpoint / f"Seed_{seed}-{class_name}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": str(manifest_path), "compact_csv": str(compact)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.destination, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
