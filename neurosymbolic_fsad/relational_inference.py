"""LLM-free runtime inference for a compiled relational rule library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cable_relations import CableRelationalRuleEngine
from .inpformer_adapter import DEFAULT_INPFORMER_ROOT, INPFormerAdapter, file_sha256


def run(args):
    engine = CableRelationalRuleEngine.load(args.rule_library)
    provenance = engine.library.get("provenance", {})
    skip_provenance = bool(
        getattr(args, "skip_provenance_check", False)
        or getattr(args, "skip_checkpoint_check", False)
    )
    expected_hash = provenance.get("checkpoint_sha256")
    actual_hash = file_sha256(args.checkpoint)
    if expected_hash and expected_hash != actual_hash and not skip_provenance:
        raise ValueError(
            "The checkpoint does not match the one used to compile the rule library"
        )
    expected_adapter = provenance.get("adapter_config")
    actual_adapter = {
        "inpformer_root": str(Path(args.inpformer_root).expanduser().resolve()),
        "encoder": args.encoder,
        "inp_num": int(args.inp_num),
        "input_size": int(args.input_size),
        "crop_size": int(args.crop_size),
        "map_size": int(args.map_size),
    }
    if expected_adapter is None and not skip_provenance:
        raise ValueError(
            "Rule library lacks adapter_config provenance; use a newly compiled "
            "library or pass --skip-provenance-check explicitly"
        )
    mismatches = {}
    if expected_adapter is not None:
        mismatches = {
            key: {"expected": expected_adapter.get(key), "actual": actual_adapter[key]}
            for key in actual_adapter
            if expected_adapter.get(key) != actual_adapter[key]
        }
    if mismatches and not skip_provenance:
        raise ValueError(
            "INP-Former adapter config differs from rule-library provenance: "
            + json.dumps(mismatches, sort_keys=True)
        )
    paths = [Path(item).expanduser().resolve() for item in args.images]
    adapter = INPFormerAdapter(
        checkpoint=args.checkpoint,
        device=args.device,
        inpformer_root=args.inpformer_root,
        encoder_name=args.encoder,
        inp_num=args.inp_num,
        input_size=args.input_size,
        crop_size=args.crop_size,
        map_size=args.map_size,
    )
    predictions = adapter.infer(paths, batch_size=args.batch_size)
    adapter.close()
    rows = []
    for path, baseline in zip(paths, predictions["scores"]):
        relation, evidence, facts, scene = engine.evaluate(path)
        rows.append({
            "path": str(path),
            "baseline_score": float(baseline),
            "relational_score": relation,
            "decision_score": engine.decision_score(float(baseline), relation),
            "top_evidence": evidence[0] if evidence else None,
            "facts": facts,
            "scene": scene,
        })
    result = {
        "schema_version": "3.0-relational-inference",
        "rule_library": str(Path(args.rule_library).expanduser().resolve()),
        "runtime_loaded_llm": False,
        "provenance_check_skipped": skip_provenance,
        "checkpoint_sha256_verified": bool(expected_hash == actual_hash),
        "adapter_config_verified": bool(expected_adapter is not None and not mismatches),
        "predictions": rows,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply a compiled relational FSAD library without loading Qwen"
    )
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-checkpoint-check", action="store_true")
    parser.add_argument(
        "--skip-provenance-check",
        action="store_true",
        help="Explicitly allow a checkpoint or adapter configuration mismatch",
    )
    return parser


def main():
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
