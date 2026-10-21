#!/usr/bin/env python3
"""Generate per-support Qwen v7 logic candidates, safely resumable by shard.

Example dual-GPU launch (the two processes may run concurrently)::

  CUDA_VISIBLE_DEVICES=0 python scripts/generate_logic_rule_candidates.py \
    --data-root /path/to/MVTec --output-root results_v7 --shot 30 \
    --num-shards 2 --shard-index 0 --llm-device cuda:0
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_logic_rule_candidates.py \
    --data-root /path/to/MVTec --output-root results_v7 --shot 30 \
    --num-shards 2 --shard-index 1 --llm-device cuda:0

Each image has its own result file.  A completed result is reused with
``--resume`` (the default); an error result is retried.  The last shard to make
all support results available also writes ``candidate_rule_bank.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.data import select_inpformer_support
from symbolicAD.qwen_logic_candidates import (
    SCHEMA_VERSION,
    QwenLogicCandidateProposer,
    aggregate_candidate_proposals,
    validate_logic_candidate,
)


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


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "image"
    return stem[:80]


def _sample_id(index: int, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"s{index:04d}_{_safe_stem(path)}_{digest}"


def _manifest_payload(
    args: argparse.Namespace, support: Sequence[Path]
) -> Dict[str, Any]:
    entries = [
        {
            "sample_index": index,
            "sample_id": _sample_id(index, path),
            "image_path": str(path.resolve()),
        }
        for index, path in enumerate(support)
    ]
    identity = {
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": len(support),
        "support_seed": args.support_seed,
        "support_strategy": "inp_official",
        "support": entries,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "support_fingerprint": fingerprint,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _is_resumable_success(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        if payload.get("status") != "ok":
            return False
        if payload.get("sample_id") != expected["sample_id"]:
            return False
        if Path(str(payload.get("image_path", ""))).resolve() != Path(
            str(expected["image_path"])
        ).resolve():
            return False
        validate_logic_candidate(payload["proposal"])
        return True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def _collect_successes(
    per_support_dir: Path, manifest: Mapping[str, Any]
) -> Tuple[List[Mapping[str, Any]], List[str], List[Dict[str, Any]]]:
    proposals: List[Mapping[str, Any]] = []
    sample_ids: List[str] = []
    missing: List[Dict[str, Any]] = []
    for entry in manifest["support"]:
        result_path = per_support_dir / f"{entry['sample_id']}.json"
        if not _is_resumable_success(result_path, entry):
            missing.append({**entry, "result_path": str(result_path)})
            continue
        payload = _read_json(result_path)
        proposals.append(validate_logic_candidate(payload["proposal"]))
        sample_ids.append(str(entry["sample_id"]))
    return proposals, sample_ids, missing


def _write_aggregate(
    run_dir: Path,
    manifest: Mapping[str, Any],
    shard_index: int,
    num_shards: int,
) -> Dict[str, Any]:
    proposals, sample_ids, missing = _collect_successes(
        run_dir / "per_support", manifest
    )
    bank = aggregate_candidate_proposals(
        proposals,
        sample_ids=sample_ids,
        expected_support_count=int(manifest["shot"]),
        class_name=str(manifest["class_name"]),
    )
    bank.update({
        "dataset": manifest["dataset"],
        "support_seed": manifest["support_seed"],
        "support_fingerprint": manifest["support_fingerprint"],
        "support": manifest["support"],
        "missing_supports": missing,
    })
    shard_bank = run_dir / (
        f"candidate_rule_bank.shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )
    _atomic_dump(shard_bank, bank)
    # Always replace the canonical bank, including with complete=false.  This
    # prevents a stale older complete bank from surviving after a support file
    # becomes missing or invalid.
    _atomic_dump(run_dir / "candidate_rule_bank.json", bank)
    return bank


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent Qwen v7 transistor rule-candidate generation"
    )
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", default="transistor")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument(
        "--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"
    )
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="reuse strictly valid per-support success files (default: true)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="do not load Qwen; only rebuild the bank from existing sample files",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.class_name.lower() != "transistor":
        raise ValueError(
            "The v7 DSL is transistor-specific; --class-name must be 'transistor'"
        )
    if args.shot < 1:
        raise ValueError("--shot must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    support = select_inpformer_support(
        args.data_root,
        args.dataset,
        args.class_name,
        args.shot,
        args.support_seed,
    )
    support = [Path(path).expanduser().resolve() for path in support]
    run_dir = (
        Path(args.output_root).expanduser().resolve()
        / args.dataset
        / args.class_name
        / f"k{args.shot}"
        / "qwen_logic_candidates_v7"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    per_support_dir = run_dir / "per_support"
    per_support_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest_payload(args, support)
    manifest_path = run_dir / "support_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("support_fingerprint") != manifest["support_fingerprint"]:
            raise RuntimeError(
                f"Refusing to mix a different support selection in {run_dir}"
            )
    _atomic_dump(manifest_path, manifest)

    assigned = [
        entry for entry in manifest["support"]
        if int(entry["sample_index"]) % args.num_shards == args.shard_index
    ]
    pending = []
    resumed = []
    for entry in assigned:
        result_path = per_support_dir / f"{entry['sample_id']}.json"
        if args.resume and _is_resumable_success(result_path, entry):
            resumed.append(str(entry["sample_id"]))
        else:
            pending.append(entry)

    failures: List[Dict[str, Any]] = []
    completed: List[str] = []
    proposer = None
    if pending and not args.aggregate_only:
        proposer = QwenLogicCandidateProposer(
            model_path=args.llm_model,
            device=args.llm_device,
            max_new_tokens=args.max_new_tokens,
            max_validation_attempts=args.validation_attempts,
        )
    if not args.aggregate_only:
        for entry in pending:
            result_path = per_support_dir / f"{entry['sample_id']}.json"
            try:
                proposal = proposer.propose(
                    entry["image_path"], class_name=args.class_name
                )
                raw_output = proposal.pop("raw_output")
                model = proposal.pop("model")
                validation_attempts_used = proposal.pop("validation_attempts_used")
                proposal.pop("schema_version", None)
                proposal.pop("class_name", None)
                # Validate the exact persisted proposal once more, independent
                # of the model wrapper's metadata.
                proposal = validate_logic_candidate(proposal)
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ok",
                    "dataset": args.dataset,
                    "class_name": args.class_name,
                    "sample_index": entry["sample_index"],
                    "sample_id": entry["sample_id"],
                    "image_path": entry["image_path"],
                    "model": model,
                    "validation_attempts_used": validation_attempts_used,
                    "proposal": proposal,
                    "raw_output": raw_output,
                }
                _atomic_dump(result_path, payload)
                completed.append(str(entry["sample_id"]))
                print(json.dumps({
                    "status": "ok",
                    "sample_id": entry["sample_id"],
                    "rule_count": len(proposal["rules"]),
                    "result": str(result_path),
                }, ensure_ascii=False), flush=True)
            except Exception as exc:  # retain progress; return nonzero after shard
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "dataset": args.dataset,
                    "class_name": args.class_name,
                    "sample_index": entry["sample_index"],
                    "sample_id": entry["sample_id"],
                    "image_path": entry["image_path"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_output": (
                        proposer.last_raw_output if proposer is not None else ""
                    ),
                }
                _atomic_dump(result_path, payload)
                failures.append(payload)
                print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    bank = _write_aggregate(
        run_dir, manifest, args.shard_index, args.num_shards
    )
    shard_summary = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_count": len(assigned),
        "resumed": resumed,
        "completed": completed,
        "failures": failures,
        "aggregate_only": args.aggregate_only,
        "bank_complete": bank["complete"],
        "successful_support_count": bank["successful_support_count"],
        "expected_support_count": bank["expected_support_count"],
        "candidate_count": bank["candidate_count"],
    }
    _atomic_dump(
        run_dir / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
        shard_summary,
    )
    print(json.dumps(shard_summary, ensure_ascii=False), flush=True)
    # A generation shard is allowed to finish before its peers.  The explicit
    # aggregate-only barrier used by launchers must, however, fail closed when
    # even one support is absent.
    return 1 if failures or (args.aggregate_only and not bank["complete"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
