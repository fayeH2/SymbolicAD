#!/usr/bin/env python3
"""LLM-free inference from a serialized category-specific normality library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurosymbolic_fsad.class_specific_rules import ClassSpecificRuleEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--baseline-scores", nargs="*", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    library_path = Path(args.rule_library).expanduser().resolve()
    library = json.loads(library_path.read_text(encoding="utf-8"))
    if library.get("runtime_requires_llm") is not False:
        raise ValueError("rule library does not explicitly guarantee LLM-free runtime")
    if args.baseline_scores is not None and len(args.baseline_scores) not in {0, len(args.images)}:
        raise ValueError("--baseline-scores must be omitted or match --images")
    engine = ClassSpecificRuleEngine(library)
    alpha = float(library.get("fusion", {}).get("alpha", 0.0))
    rows = []
    for index, image in enumerate(args.images):
        relation, evidence, facts, diagnostics = engine.evaluate(image)
        baseline = None if not args.baseline_scores else float(args.baseline_scores[index])
        rows.append({
            "path": str(Path(image).expanduser().resolve()),
            "baseline_score": baseline,
            "relation_score": relation,
            "decision_score": None if baseline is None else baseline + alpha * relation,
            "facts": facts,
            "diagnostics": diagnostics,
            "evidence": evidence,
        })
    payload = {
        "schema_version": "5.0-class-specific-static-inference",
        "rule_library": str(library_path),
        "runtime_loaded_llm": False,
        "dataset": library["dataset"],
        "class_name": library["class_name"],
        "predictions": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
