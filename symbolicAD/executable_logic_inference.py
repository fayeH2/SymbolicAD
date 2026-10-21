"""LLM-free deployment for a compiled v7 executable-logic library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .inpformer_adapter import INPFormerAdapter, file_sha256
from .transistor_logic import (
    TransistorLogicEngine,
    execute_grounded_neural_symbolic,
    preprocess_rgb,
)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _overlay(image: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(
        np.rint(np.clip(score_map, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.52, heat, 0.48, 0.0)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(
        result, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    library_path = Path(args.rule_library).expanduser().resolve()
    library = json.loads(library_path.read_text(encoding="utf-8"))
    engine = TransistorLogicEngine(library)
    baseline = library.get("baseline", {})
    config = baseline.get("adapter_config")
    if not isinstance(config, dict) or any(config.get(key) is None for key in (
        "inpformer_root", "encoder", "inp_num", "input_size", "crop_size", "map_size"
    )):
        raise ValueError("Rule library lacks a complete frozen adapter_config")
    checkpoint = Path(args.checkpoint or baseline.get("checkpoint", "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"INP-Former checkpoint missing: {checkpoint}")
    actual_hash = file_sha256(checkpoint)
    expected_hash = baseline.get("checkpoint_sha256")
    if expected_hash and actual_hash != expected_hash and not args.skip_checkpoint_check:
        raise ValueError("Checkpoint SHA-256 differs from the compiled rule library")

    paths = [Path(path).expanduser().resolve() for path in args.images]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Query images missing: {missing}")
    adapter = INPFormerAdapter(
        checkpoint=checkpoint,
        device=args.device,
        inpformer_root=config["inpformer_root"],
        encoder_name=config["encoder"],
        inp_num=int(config["inp_num"]),
        input_size=int(config["input_size"]),
        crop_size=int(config["crop_size"]),
        map_size=int(config["map_size"]),
    )
    neural = adapter.infer(paths, batch_size=args.batch_size)
    adapter.close()
    baseline_scores = np.clip(neural["scores"], 0.0, 1.0)
    baseline_maps = np.clip(neural["maps"], 0.0, 1.0)
    symbolic_maps = []
    fused_maps = []
    rows = []
    output = Path(args.output_dir).expanduser().resolve()
    visuals = output / "visualizations"
    visuals.mkdir(parents=True, exist_ok=True)
    for index, (path, baseline_score, baseline_map) in enumerate(
        zip(paths, baseline_scores, baseline_maps)
    ):
        inference = engine.evaluate(path)
        final_score, final_trace = execute_grounded_neural_symbolic(
            float(baseline_score), inference.symbolic_anomaly
        )
        final_map = np.asarray(execute_grounded_neural_symbolic(
            baseline_map, inference.symbolic_map
        )[0], dtype=np.float32)
        violated = [
            rule_id for rule_id, evidence in inference.atoms.items()
            if float(evidence["violation"]) > 1e-8
        ]
        rows.append({
            "path": str(path),
            "baseline_score": float(baseline_score),
            "symbolic_score": inference.symbolic_anomaly,
            "neuro_symbolic_score": float(final_score),
            "violated_rules": violated,
            "explanation": inference.explanation,
            "facts": inference.facts,
            "rule_atoms": inference.atoms,
            "symbolic_logic_nodes": inference.scalar_trace,
            "final_fusion_logic_nodes": final_trace,
            "diagnostics": inference.diagnostics,
        })
        symbolic_maps.append(inference.symbolic_map)
        fused_maps.append(final_map)
        original = preprocess_rgb(path)
        comparison = np.concatenate([
            _label(original, "Query"),
            _label(_overlay(original, baseline_map), "INP-Former"),
            _label(_overlay(original, inference.symbolic_map), "Rule violations"),
            _label(_overlay(original, final_map), "Executable neuro-symbolic"),
        ], axis=1)
        Image.fromarray(comparison).save(
            visuals / f"{index:03d}_{path.stem}_comparison.jpg", quality=95
        )
    result = {
        "schema_version": "7.0-llm-free-executable-logic-inference",
        "rule_library": str(library_path),
        "runtime_loaded_llm": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_verified": bool(expected_hash == actual_hash),
        "logic_formula": library["logic"]["formula"],
        "accepted_rules": library["accepted_rule_ids"],
        "predictions": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions_v7_static_inference.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "maps_v7_static_inference.npz",
        paths=np.asarray([str(path) for path in paths]),
        baseline_maps=baseline_maps,
        symbolic_maps=np.asarray(symbolic_maps, dtype=np.float32),
        fused_maps=np.asarray(fused_maps, dtype=np.float32),
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a compiled v7 rule library without loading Qwen"
    )
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-checkpoint-check", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
