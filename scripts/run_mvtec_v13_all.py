#!/usr/bin/env python3
"""Run frozen V13 over all MVTec classes with one worker per GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def proposal_path(root: Path, class_name: str) -> Path:
    special = {
        "cable": root / "results_v12_fresh_proposals/mvtec/cable/k4/qwen_generic_relational_proposal.json",
        "transistor": root / "results_v11_fresh_proposals/mvtec/transistor/k4/qwen_generic_relational_proposal.json",
    }
    return special.get(
        class_name,
        root / f"results_v13_fresh_proposals/mvtec/{class_name}/k4/qwen_generic_relational_proposal.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--output-root", default="results_v13_mvtec_all")
    parser.add_argument("--classes", nargs="*", default=CLASSES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"; log_root.mkdir(exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    checkpoint = (
        "/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/"
        "INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_"
        "Resize=448_Crop=392_INP_num=6/model.pth"
    )

    def worker(gpu: str, class_names: list[str]) -> list[dict]:
        rows = []
        for class_name in class_names:
            proposal = proposal_path(root, class_name)
            output = output_root / class_name / "k4"
            command = [
                sys.executable, str(root / "run_entity_bridge_fsad.py"),
                "--dataset", "mvtec",
                "--data-root", "/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec",
                "--class-name", class_name, "--shot", "4", "--support-seed", "1",
                "--proposal-json", str(proposal), "--checkpoint", checkpoint,
                "--inpformer-root", "/home/hjh/Desktop/H2VLR_compare/INP-Former",
                "--inp-device", f"cuda:{gpu}", "--ovd-device", f"cuda:{gpu}",
                "--batch-size", "12", "--max-visualizations", "10",
                "--output-dir", str(output),
            ]
            log_path = log_root / f"{class_name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                    text=True, check=False,
                )
            metric_path = output / "metrics_v13.json"
            row = {
                "class_name": class_name, "gpu": gpu,
                "returncode": process.returncode, "log": str(log_path),
            }
            if metric_path.is_file():
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                row.update({
                    "bridge_abstained": metric["bridge_abstained"],
                    "baseline": metric["baseline"],
                    "fusion": metric["entity_bridge_fusion"],
                    "delta": metric["delta_fused_minus_baseline"],
                    "trigger_rate_normal": metric["trigger_rate_normal"],
                    "trigger_rate_anomaly": metric["trigger_rate_anomaly"],
                })
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return rows

    assignments = [args.classes[index::len(gpus)] for index in range(len(gpus))]
    all_rows = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, classes) for gpu, classes in zip(gpus, assignments)]
        for future in as_completed(futures):
            all_rows.extend(future.result())
    all_rows.sort(key=lambda row: CLASSES.index(row["class_name"]))
    (output_root / "mvtec_v13_summary.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if any(row["returncode"] != 0 for row in all_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
