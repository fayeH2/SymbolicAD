"""Online inference using an already-built static rule library (no LLM)."""

import argparse
import json
from pathlib import Path

from .features import extract_features
from .inpformer_adapter import DEFAULT_INPFORMER_ROOT, INPFormerAdapter, file_sha256
from .reliable_rules import ReliableRuleEngine
from .rule_engine import StaticRuleEngine


def main():
    parser = argparse.ArgumentParser(description="INP-Former + static normality-rule inference")
    parser.add_argument("--rules", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--skip-checkpoint-verification", action="store_true")
    args = parser.parse_args()

    paths = [Path(path).expanduser().resolve() for path in args.images]
    library = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    is_v2 = library.get("schema_version") == "2.0"
    engine = ReliableRuleEngine(library) if is_v2 else StaticRuleEngine(library)
    if is_v2 and not args.skip_checkpoint_verification:
        expected_hash = library.get("baseline", {}).get("checkpoint_sha256")
        if expected_hash and file_sha256(args.checkpoint) != expected_hash:
            raise ValueError("Checkpoint SHA256 does not match the v2 rule library")
    adapter = INPFormerAdapter(
        checkpoint=args.checkpoint,
        device=args.device,
        inpformer_root=args.inpformer_root,
        encoder_name=args.encoder,
        inp_num=args.inp_num,
    )
    predictions = adapter.infer(paths, batch_size=args.batch_size)
    adapter.close()
    rows = []
    for path, anomaly_map, baseline in zip(
        paths, predictions["maps"], predictions["scores"]
    ):
        if is_v2:
            facts = engine.facts(path, anomaly_map)
            decision, rule_score, evidence = engine.decision_score(float(baseline), facts)
            rows.append({
                "path": str(path), "baseline_score": float(baseline),
                "baseline_z": engine.baseline_z(float(baseline)),
                "rule_score": rule_score, "decision_score": decision,
                "fusion_enabled": bool(engine.fusion["enabled"]),
                "triggered_rules": [item for item in evidence if item["violation"] > 0],
            })
        else:
            facts = extract_features(path, anomaly_map)
            rule_score, evidence = engine.score(facts)
            rows.append({
                "path": str(path),
                "baseline_score": float(baseline),
                "baseline_normalized": engine.normalize_baseline(float(baseline)),
                "rule_score": rule_score,
                "fused_score": engine.fused_score(float(baseline), rule_score, args.alpha),
                "triggered_rules": [item for item in evidence if item["violation"] > 0],
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
