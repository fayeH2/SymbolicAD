"""LLM-free inference for a compiled generic v4 relation library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generic_relations import GenericRelationalRuleEngine
from .inpformer_adapter import DEFAULT_INPFORMER_ROOT, INPFormerAdapter, file_sha256


def run(args):
    engine = GenericRelationalRuleEngine.load(args.rule_library)
    expected = engine.library.get("inpformer", {})
    actual_hash = file_sha256(args.checkpoint)
    if expected.get("checkpoint_sha256") != actual_hash and not args.skip_provenance_check:
        raise ValueError("Checkpoint SHA-256 differs from the compiled rule library")
    actual = {
        "inpformer_root": str(Path(args.inpformer_root).expanduser().resolve()),
        "encoder": args.encoder, "inp_num": args.inp_num,
        "input_size": args.input_size, "crop_size": args.crop_size,
        "map_size": args.map_size,
    }
    mismatches = {key: {"expected": expected.get(key), "actual": value}
                  for key, value in actual.items() if expected.get(key) != value}
    if mismatches and not args.skip_provenance_check:
        raise ValueError("Adapter provenance mismatch: " + json.dumps(mismatches))
    paths = [Path(item).expanduser().resolve() for item in args.images]
    adapter = INPFormerAdapter(
        checkpoint=args.checkpoint, device=args.device,
        inpformer_root=args.inpformer_root, encoder_name=args.encoder,
        inp_num=args.inp_num, input_size=args.input_size,
        crop_size=args.crop_size, map_size=args.map_size,
    )
    predictions = adapter.infer(paths, batch_size=args.batch_size)
    adapter.close()
    fusion = engine.library.get("fusion", {})
    center = float(fusion.get("baseline_center", 0.0))
    scale = max(float(fusion.get("baseline_scale", 1.0)), 1e-8)
    rows = []
    for path, baseline in zip(paths, predictions["scores"]):
        relation, evidence, facts, diagnostics = engine.evaluate(path)
        baseline_z = (float(baseline) - center) / scale
        decision = float(baseline) + float(fusion.get("alpha", 0.0)) * relation \
            if fusion.get("enabled", False) else float(baseline)
        rows.append({
            "path": str(path), "baseline_score": float(baseline),
            "relation_score": relation, "decision_score": decision,
            "top_evidence": evidence[0] if evidence else None,
            "facts": facts, "diagnostics": diagnostics,
        })
    result = {
        "schema_version": "4.0-generic-relational-inference",
        "rule_library": str(Path(args.rule_library).resolve()),
        "runtime_loaded_llm": False,
        "checkpoint_sha256_verified": expected.get("checkpoint_sha256") == actual_hash,
        "adapter_config_verified": not mismatches,
        "predictions": rows,
    }
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--skip-provenance-check", action="store_true")
    return parser


def main():
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
