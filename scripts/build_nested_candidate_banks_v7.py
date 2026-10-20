#!/usr/bin/env python3
"""Build k-prefix Qwen banks from one frozen larger support ordering.

No model is loaded.  Every prefix reuses the exact per-image Qwen response
generated for the largest support set, making k4 ⊂ k8 ⊂ k16 ⊂ k30 a genuine
sample-size control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from neurosymbolic_fsad.qwen_logic_candidates import (
    SCHEMA_VERSION,
    aggregate_candidate_proposals,
    validate_logic_candidate,
)


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _atomic_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fingerprint(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_prefixes(
    source_run_dir: Path, output_root: Path, counts: list[int]
) -> list[dict[str, Any]]:
    source_run_dir = source_run_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source_manifest_path = source_run_dir / "support_manifest.json"
    source_manifest = _read(source_manifest_path)
    entries = sorted(
        source_manifest["support"], key=lambda row: int(row["sample_index"])
    )
    if not counts or min(counts) < 4 or max(counts) > len(entries):
        raise ValueError(
            f"counts must be in [4, {len(entries)}], received {counts}"
        )
    outputs = []
    for count in sorted(set(counts)):
        selected = [dict(row) for row in entries[:count]]
        proposals = []
        sample_ids = []
        source_results = []
        for entry in selected:
            result_path = source_run_dir / "per_support" / f"{entry['sample_id']}.json"
            result = _read(result_path)
            if result.get("status") != "ok":
                raise ValueError(f"Source result is not valid: {result_path}")
            if result.get("sample_id") != entry["sample_id"]:
                raise ValueError(f"Source result ID mismatch: {result_path}")
            if Path(str(result.get("image_path", ""))).resolve() != Path(
                str(entry["image_path"])
            ).resolve():
                raise ValueError(f"Source result image-path mismatch: {result_path}")
            proposal = validate_logic_candidate(result["proposal"])
            proposals.append(proposal)
            sample_ids.append(str(entry["sample_id"]))
            source_results.append(str(result_path))
        identity = {
            "dataset": source_manifest["dataset"],
            "class_name": source_manifest["class_name"],
            "shot": count,
            "support_seed": source_manifest["support_seed"],
            "support_strategy": "nested_prefix_of_frozen_larger_manifest",
            "support": selected,
        }
        fingerprint = _fingerprint(identity)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "support_fingerprint": fingerprint,
            "source_manifest": str(source_manifest_path),
            "source_support_count": len(entries),
            "strict_nested_prefix": True,
        }
        bank = aggregate_candidate_proposals(
            proposals,
            sample_ids=sample_ids,
            expected_support_count=count,
            class_name=str(source_manifest["class_name"]),
        )
        bank.update({
            "dataset": source_manifest["dataset"],
            "support_seed": source_manifest["support_seed"],
            "support_fingerprint": fingerprint,
            "support": selected,
            "missing_supports": [],
            "source_manifest": str(source_manifest_path),
            "source_results": source_results,
            "strict_nested_prefix": True,
        })
        run_dir = (
            output_root / str(source_manifest["dataset"])
            / str(source_manifest["class_name"]) / f"nested_k{count}"
            / "qwen_logic_candidates_v7"
        )
        _atomic_dump(run_dir / "support_manifest.json", manifest)
        _atomic_dump(run_dir / "candidate_rule_bank.json", bank)
        outputs.append({
            "support_count": count,
            "candidate_count": bank["candidate_count"],
            "manifest": str(run_dir / "support_manifest.json"),
            "candidate_bank": str(run_dir / "candidate_rule_bank.json"),
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 8, 16, 30])
    args = parser.parse_args()
    outputs = build_prefixes(args.source_run_dir, args.output_root, args.counts)
    print(json.dumps({"prefixes": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
