#!/usr/bin/env python3
"""Multi-GPU launcher for baseline training, rule induction and evaluation.

Every subprocess sees exactly one physical GPU as ``cuda:0``.  A dynamic job
queue keeps all requested GPUs busy while preserving one-Qwen-process-per-GPU
as the safe default for 24 GB RTX 3090 cards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.data import IMAGE_SUFFIXES, select_inpformer_support


DEFAULT_CONFIG = PROJECT / "configs" / "multidataset_v6.json"
CLASS_SPECIFIC = {
    ("mvtec", "transistor"),
    ("visa", "pcb1"),
    ("visa", "macaroni2"),
    ("mpdd", "bracket_brown"),
}


@dataclass
class Job:
    name: str
    command: list[str]
    log_path: Path


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def _recursive_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def _discover_classes(entry: dict[str, Any]) -> list[str]:
    classes = entry.get("classes", [])
    if classes != "discover_mvt_like":
        return list(classes)
    root = Path(entry["data_root"])
    if not root.is_dir():
        return []
    test_subdir = entry.get("test_subdir", "test")
    found = []
    for child in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda p: p.name):
        normal = _images(child / "train" / "good") or _images(child / "train" / "ok")
        if normal and (child / test_subdir).is_dir():
            found.append(child.name)
    return found


def _load_registry(path: Path, shot: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for source in payload["datasets"]:
        entry = dict(source)
        entry["classes"] = _discover_classes(entry)
        if "checkpoint" in entry:
            entry["checkpoint"] = entry["checkpoint"].format(shot=shot)
        entries.append(entry)
    return payload, entries


def _scope_entries(entries, scope: str, dataset_ids: list[str] | None):
    if dataset_ids:
        wanted = set(dataset_ids)
        selected = [entry for entry in entries if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in selected}
        if missing:
            raise ValueError(f"Unknown dataset ids: {sorted(missing)}")
        return selected
    if scope == "all":
        return list(entries)
    return [entry for entry in entries if entry["domain"] == scope]


def _preflight(entries, args, registry) -> dict[str, Any]:
    rows, errors, warnings = [], [], []
    defaults = registry["defaults"]
    model = Path(args.llm_model or defaults["llm_model"]).expanduser().resolve()
    inp_root = Path(defaults["inpformer_root"]).expanduser().resolve()
    if args.stage in {"all", "propose"} and not model.is_dir():
        errors.append(f"Local Qwen model is missing: {model}")
    if args.stage in {"all", "baseline", "evaluate"} and not inp_root.is_dir():
        errors.append(f"INP-Former repository is missing: {inp_root}")

    for entry in entries:
        row = {
            "dataset": entry["id"],
            "domain": entry["domain"],
            "status": entry["status"],
            "data_root": entry["data_root"],
            "classes": len(entry.get("classes", [])),
            "ready": False,
            "issues": [],
        }
        root = Path(entry["data_root"])
        if not root.exists():
            row["issues"].append("data_root_missing")
        if entry["status"] == "incompatible":
            row["issues"].append(entry.get("reason", "incompatible layout"))
            warnings.append(f"{entry['id']}: {entry.get('reason', 'incompatible')}")
            rows.append(row)
            continue
        if entry["status"] == "experimental_trainable" and not args.include_experimental:
            row["issues"].append("requires --include-experimental")
            rows.append(row)
            continue
        if not entry["classes"]:
            row["issues"].append("no compatible classes discovered")
        test_subdir = entry.get("test_subdir", "test")
        for class_name in entry["classes"]:
            class_root = root / class_name
            normal = _images(class_root / "train" / "good") or _images(class_root / "train" / "ok")
            test_root = class_root / test_subdir
            if len(normal) < args.shot:
                row["issues"].append(f"{class_name}: only {len(normal)} normal train images")
            if not test_root.is_dir():
                row["issues"].append(f"{class_name}: missing {test_subdir}")
            else:
                test_dirs = [folder for folder in test_root.iterdir() if folder.is_dir()]
                normal_count = sum(
                    len(_recursive_images(folder)) for folder in test_dirs
                    if folder.name.lower() in {"good", "ok"}
                )
                anomaly_count = sum(
                    len(_recursive_images(folder)) for folder in test_dirs
                    if folder.name.lower() not in {"good", "ok", "ground_truth", "masks", "mask"}
                )
                if not normal_count:
                    row["issues"].append(f"{class_name}: no normal test images")
                if not anomaly_count:
                    row["issues"].append(f"{class_name}: no anomaly test images")
            try:
                select_inpformer_support(
                    root, entry["id"], class_name, args.shot, args.support_seed
                )
            except Exception as exc:
                row["issues"].append(f"{class_name}: support selection: {exc}")
        checkpoint = Path(entry["checkpoint"])
        row["checkpoint"] = str(checkpoint)
        row["checkpoint_exists"] = checkpoint.is_file()
        will_train = args.stage in {"all", "baseline"}
        if args.stage == "evaluate" and not checkpoint.is_file():
            row["issues"].append("checkpoint missing; run --stage baseline first")
        if args.stage == "evaluate":
            for class_name in entry["classes"]:
                proposal = (
                    Path(args.output_root) / "proposals" / entry["id"] / class_name /
                    f"k{args.shot}" / "qwen_generic_relational_proposal.json"
                )
                if not proposal.is_file():
                    row["issues"].append(
                        f"{class_name}: proposal missing; run --stage propose first"
                    )
        if args.stage == "all" and not checkpoint.is_file() and not entry.get("baseline_trainer"):
            row["issues"].append("checkpoint missing and no baseline trainer registered")
        row["ready"] = not row["issues"] or (
            will_train and row["issues"] == []
        )
        rows.append(row)
        errors.extend(f"{entry['id']}: {issue}" for issue in row["issues"])

    report = {
        "schema_version": "6.0-preflight",
        "created_at_unix": time.time(),
        "scope": args.scope,
        "shot": args.shot,
        "gpus": args.gpus,
        "datasets": rows,
        "warnings": warnings,
        "errors": errors,
        "strict_inventory": args.strict_inventory,
    }
    if args.strict_inventory:
        incompatible = [row for row in rows if row["status"] == "incompatible"]
        if incompatible:
            report["errors"].append(
                "strict inventory requested but incompatible datasets exist: "
                + ", ".join(row["dataset"] for row in incompatible)
            )
    return report


def _gpu_ids(text: str) -> list[str]:
    if text != "auto":
        result = [token.strip() for token in text.split(",") if token.strip()]
        if not result:
            raise ValueError("--gpus resolved to an empty list")
        return result
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [token.strip() for token in visible.split(",") if token.strip()]
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        result = [line.strip() for line in output.splitlines() if line.strip()]
        if result:
            return result
    except (OSError, subprocess.CalledProcessError):
        pass
    raise RuntimeError("No GPUs detected; pass --gpus explicitly")


def _run_jobs(jobs: list[Job], gpus: list[str], per_gpu: int, stage: str, dry_run: bool):
    if not jobs:
        print(f"[{stage}] no pending jobs", flush=True)
        return []
    slots = [gpu for gpu in gpus for _ in range(per_gpu)]
    print(
        f"[{stage}] {len(jobs)} jobs, GPUs={gpus}, workers_per_gpu={per_gpu}",
        flush=True,
    )
    if dry_run:
        for index, job in enumerate(jobs):
            gpu = slots[index % len(slots)]
            print(f"[dry-run][gpu={gpu}] {job.name}: {shlex.join(job.command)}")
        return []

    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    results, lock = [], threading.Lock()

    def worker(gpu: str, worker_id: int):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": str(max(1, args.cpu_threads_per_job)),
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            print(f"[{stage}][gpu={gpu}][start] {job.name}", flush=True)
            started = time.time()
            with job.log_path.open("w", encoding="utf-8") as handle:
                handle.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n# {shlex.join(job.command)}\n")
                handle.flush()
                completed = subprocess.run(
                    job.command,
                    cwd=str(PROJECT),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            row = {
                "stage": stage,
                "job": job.name,
                "gpu": gpu,
                "worker_id": worker_id,
                "returncode": completed.returncode,
                "seconds": round(time.time() - started, 2),
                "log": str(job.log_path),
            }
            with lock:
                results.append(row)
            print(
                f"[{stage}][gpu={gpu}][done rc={completed.returncode}] "
                f"{job.name} ({row['seconds']}s)", flush=True,
            )
            pending.task_done()

    threads = []
    for worker_id, gpu in enumerate(slots):
        thread = threading.Thread(target=worker, args=(gpu, worker_id), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return sorted(results, key=lambda row: row["job"])


def _is_cuda_oom(row: dict[str, Any]) -> bool:
    if row.get("returncode") == 0:
        return False
    path = Path(row.get("log", ""))
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _baseline_command(entry, args, defaults) -> list[str]:
    python = args.python
    inp_root = Path(defaults["inpformer_root"])
    # For official trainers the checkpoint is always <save_root>/<run>/model.pth.
    # Deriving save_root from the selected checkpoint keeps explicit retraining
    # inside this project's artifacts directory.
    save_root = Path(entry["checkpoint"]).parent.parent
    kind = entry["baseline_trainer"]
    common_epochs = str(args.baseline_epochs)
    if kind == "official_few_shot":
        return [
            python, str(inp_root / "INP_Former_Few_Shot.py"),
            "--dataset", entry["trainer_dataset"],
            "--data_path", entry["data_root"],
            "--save_dir", str(save_root),
            "--shot", str(args.shot),
            "--total_epochs", common_epochs,
            "--batch_size", str(args.batch_size),
            "--phase", "train",
        ]
    if kind == "official_industrial":
        return [
            python, str(inp_root / "INP_Former_Industrial.py"),
            "--dataset", entry["id"],
            "--data-root", str(Path(entry["data_root"]).parent),
            "--save-dir", str(save_root),
            "--output-root", str(Path(args.output_root) / "baseline_metrics"),
            "--shot", str(args.shot),
            "--total-epochs", common_epochs,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.data_workers),
            "--device", "cuda:0",
            "--no-visual",
        ]
    if kind == "official_medical":
        return [
            python, str(inp_root / "INP_Former_Medical.py"),
            "--dataset", entry["id"],
            "--save-dir", str(save_root),
            "--output-root", str(Path(args.output_root) / "baseline_metrics"),
            "--shot", str(args.shot),
            "--total-epochs", common_epochs,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.data_workers),
            "--device", "cuda:0",
        ]
    if kind == "isolated_generic":
        command = [
            python, str(PROJECT / "scripts" / "train_generic_inpformer.py"),
            "--dataset", entry["id"],
            "--data-root", entry["data_root"],
            "--classes", *entry["classes"],
            "--checkpoint", entry["checkpoint"],
            "--inpformer-root", str(inp_root),
            "--shot", str(args.shot),
            "--support-seed", str(args.support_seed),
            "--epochs", common_epochs,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.data_workers),
            "--device", "cuda:0",
        ]
        if args.retrain_baseline:
            command.append("--force")
        return command
    raise ValueError(f"Unsupported baseline trainer: {kind}")


def _baseline_jobs(entries, args, defaults):
    jobs = []
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if checkpoint.is_file() and not args.retrain_baseline:
            continue
        jobs.append(Job(
            name=f"baseline.{entry['id']}.k{args.shot}",
            command=_baseline_command(entry, args, defaults),
            log_path=Path(args.output_root) / "logs" / "baseline" / f"{entry['id']}.log",
        ))
    return jobs


def _proposal_jobs(entries, args, defaults):
    jobs = []
    proposal_root = Path(args.output_root) / "proposals"
    model = args.llm_model or defaults["llm_model"]
    for entry in entries:
        classes = entry["classes"]
        shard_count = max(1, int(entry.get("proposal_shards", 1)))
        shards = [classes[index::shard_count] for index in range(shard_count)]
        for shard_index, shard in enumerate(shards):
            if not shard:
                continue
            command = [
                args.python, str(PROJECT / "scripts" / "generate_multidataset_proposals.py"),
                "--dataset", entry["id"],
                "--data-root", entry["data_root"],
                "--output-root", str(proposal_root),
                "--classes", *shard,
                "--shot", str(args.shot),
                "--support-seed", str(args.support_seed),
                "--llm-model", model,
                "--llm-device", "cuda:0",
                "--max-new-tokens", str(args.max_new_tokens),
                "--summary-name", f"proposal_summary.part-{shard_index + 1}-of-{shard_count}.json",
            ]
            if args.resume:
                command.append("--resume")
            jobs.append(Job(
                name=f"propose.{entry['id']}.{shard_index + 1}of{shard_count}",
                command=command,
                log_path=Path(args.output_root) / "logs" / "propose" /
                         f"{entry['id']}.{shard_index + 1}of{shard_count}.log",
            ))
    return jobs


def _evaluation_jobs(entries, args, defaults):
    jobs = []
    output_root = Path(args.output_root)
    for entry in entries:
        for class_name in entry["classes"]:
            output = output_root / "generic" / entry["id"] / class_name / f"k{args.shot}"
            metric = output / "metrics_v4_generic.json"
            if args.resume and metric.is_file():
                continue
            proposal = (
                output_root / "proposals" / entry["id"] / class_name / f"k{args.shot}" /
                "qwen_generic_relational_proposal.json"
            )
            cache = output_root / "cache" / entry["id"] / class_name / f"k{args.shot}"
            command = [
                args.python, str(PROJECT / "run_generic_relational_fsad.py"),
                "--dataset", entry["id"],
                "--data-root", entry["data_root"],
                "--class-name", class_name,
                "--test-subdir", entry.get("test_subdir", "test"),
                "--shot", str(args.shot),
                "--support-seed", str(args.support_seed),
                "--support-strategy", "inp_official",
                "--checkpoint", entry["checkpoint"],
                "--inpformer-root", defaults["inpformer_root"],
                "--proposal-json", str(proposal),
                "--output-dir", str(output),
                "--prediction-cache-dir", str(cache),
                "--device", "cuda:0",
                "--batch-size", str(args.batch_size),
                "--maximum-rule-fpr", str(args.maximum_rule_fpr),
                "--llm-model", args.llm_model or defaults["llm_model"],
            ]
            jobs.append(Job(
                name=f"evaluate.{entry['id']}.{class_name}.k{args.shot}",
                command=command,
                log_path=output_root / "logs" / "evaluate" /
                         f"{entry['id']}.{class_name}.log",
            ))
    return jobs


def _class_specific_jobs(entries, args, defaults):
    jobs = []
    if args.shot != 4:
        print("[class-specific] skipped: frozen v5 programs require --shot 4", flush=True)
        return jobs
    output_root = Path(args.output_root)
    for entry in entries:
        for class_name in entry["classes"]:
            if (entry["id"], class_name) not in CLASS_SPECIFIC:
                continue
            output = output_root / "class_specific" / entry["id"] / class_name / f"k{args.shot}"
            if args.resume and (output / "metrics_v5_class_specific.json").is_file():
                continue
            source = output_root / "generic" / entry["id"] / class_name / f"k{args.shot}"
            cache = output_root / "cache" / entry["id"] / class_name / f"k{args.shot}"
            command = [
                args.python, str(PROJECT / "run_class_specific_symbolicAD.py"),
                "--dataset", entry["id"],
                "--class-name", class_name,
                "--data-root", entry["data_root"],
                "--source-run-dir", str(source),
                "--prediction-cache-dir", str(cache),
                "--output-dir", str(output),
                "--llm-model", args.llm_model or defaults["llm_model"],
            ]
            jobs.append(Job(
                name=f"class-specific.{entry['id']}.{class_name}.k{args.shot}",
                command=command,
                log_path=output_root / "logs" / "class_specific" /
                         f"{entry['id']}.{class_name}.log",
            ))
    return jobs


def _export_job(entries, args):
    command = [
        args.python,
        str(PROJECT / "scripts" / "export_h2vlr_layout.py"),
        "--config", str(Path(args.config).expanduser().resolve()),
        "--output-root", str(Path(args.output_root).expanduser().resolve()),
        "--shot", str(args.shot),
        "--support-seed", str(args.support_seed),
        "--visual-size", str(args.visual_size),
        "--max-visualizations-per-class", str(args.max_visualizations_per_class),
        "--datasets", *[entry["id"] for entry in entries],
    ]
    return Job(
        name=f"export.h2vlr-layout.k{args.shot}",
        command=command,
        log_path=Path(args.output_root) / "logs" / "export.log",
    )


def _summarize(entries, args, inventory, job_results):
    output_root = Path(args.output_root)
    rows, missing = [], []
    for entry in entries:
        for class_name in entry["classes"]:
            metric_path = (
                output_root / "generic" / entry["id"] / class_name / f"k{args.shot}" /
                "metrics_v4_generic.json"
            )
            if not metric_path.is_file():
                missing.append(str(metric_path))
                continue
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            row = {
                "domain": entry["domain"],
                "dataset": entry["id"],
                "class_name": class_name,
                "baseline_i_auroc": payload["baseline"]["i_auroc"],
                "rules_i_auroc": payload["baseline_plus_rules"]["i_auroc"],
                "delta_i_auroc": payload["delta_i_auroc"],
                "baseline_i_aupr": payload["baseline"]["i_aupr"],
                "rules_i_aupr": payload["baseline_plus_rules"]["i_aupr"],
                "delta_i_aupr": payload["delta_i_aupr"],
                "rule_count": payload["selected_rule_count"],
                "fusion_enabled": payload["fusion"]["enabled"],
                "proposal_abstained": payload["proposal_abstained"],
                "generic_metrics": str(metric_path),
            }
            specific_path = (
                output_root / "class_specific" / entry["id"] / class_name /
                f"k{args.shot}" / "metrics_v5_class_specific.json"
            )
            if specific_path.is_file():
                specific = json.loads(specific_path.read_text(encoding="utf-8"))
                row.update({
                    "class_specific_i_auroc": specific["baseline_plus_rules"]["i_auroc"],
                    "class_specific_delta_i_auroc": specific["delta_i_auroc"],
                    "class_specific_metrics": str(specific_path),
                })
            rows.append(row)

    datasets = []
    for entry in entries:
        selected = [row for row in rows if row["dataset"] == entry["id"]]
        if not selected:
            continue
        datasets.append({
            "dataset": entry["id"],
            "domain": entry["domain"],
            "class_count": len(selected),
            "proposal_coverage": int(sum(not row["proposal_abstained"] for row in selected)),
            "rule_coverage": int(sum(row["rule_count"] > 0 for row in selected)),
            "fusion_enabled_classes": int(sum(row["fusion_enabled"] for row in selected)),
            "improved_classes": int(sum(row["delta_i_auroc"] > 0 for row in selected)),
            "degraded_classes": int(sum(row["delta_i_auroc"] < 0 for row in selected)),
            "macro_baseline_i_auroc": float(np.mean([row["baseline_i_auroc"] for row in selected])),
            "macro_rules_i_auroc": float(np.mean([row["rules_i_auroc"] for row in selected])),
            "macro_delta_i_auroc": float(np.mean([row["delta_i_auroc"] for row in selected])),
            "macro_baseline_i_aupr": float(np.mean([row["baseline_i_aupr"] for row in selected])),
            "macro_rules_i_aupr": float(np.mean([row["rules_i_aupr"] for row in selected])),
            "macro_delta_i_aupr": float(np.mean([row["delta_i_aupr"] for row in selected])),
        })
    summary = {
        "schema_version": "6.0-multidataset-neurosymbolic-summary",
        "shot": args.shot,
        "support_seed": args.support_seed,
        "llm_model": args.llm_model or "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking",
        "runtime_llm_usage": "library construction only; compiled inference is LLM-free",
        "test_labels_used_for_rule_selection": False,
        "dataset_count": len(datasets),
        "class_count": len(rows),
        "datasets": datasets,
        "classes": rows,
        "missing_metrics": missing,
        "inventory": inventory,
        "jobs": job_results,
        "interpretation_guardrail": (
            "A large improvement is a hypothesis to test, not an assumed property. "
            "Report rule coverage, normal trigger rate and per-class deltas together."
        ),
    }
    destination = output_root / "summary" / f"multidataset_k{args.shot}.json"
    _dump(destination, summary)
    csv_path = destination.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "summary": str(destination),
        "csv": str(csv_path),
        "datasets": len(datasets),
        "classes": len(rows),
        "missing": len(missing),
    }, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage",
        choices=("preflight", "baseline", "propose", "evaluate", "class-specific", "export", "summary", "validate", "all"),
        default="all",
    )
    parser.add_argument("--scope", choices=("industrial", "medical", "all"), default="all")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--strict-inventory", action="store_true")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--gpus", default="auto", help="Comma-separated physical GPU ids")
    parser.add_argument("--baseline-workers-per-gpu", type=int, default=1)
    parser.add_argument("--proposal-workers-per-gpu", type=int, default=1)
    parser.add_argument("--eval-workers-per-gpu", type=int, default=1)
    parser.add_argument("--class-specific-workers-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--baseline-epochs", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--maximum-rule-fpr", type=float, default=0.0)
    parser.add_argument("--visual-size", type=int, default=400)
    parser.add_argument(
        "--max-visualizations-per-class", type=int, default=0,
        help="0 saves every test image in the H2VLR-style export.",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--output-root", default=str(PROJECT / "results_v6_multidataset"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--retrain-baseline", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    global args
    args = build_parser().parse_args()
    config = Path(args.config).expanduser().resolve()
    registry, all_entries = _load_registry(config, args.shot)
    if args.retrain_baseline:
        retrained_root = PROJECT / "artifacts" / "baselines" / "retrained"
        for entry in all_entries:
            if "checkpoint" not in entry:
                continue
            original_run_name = Path(entry["checkpoint"]).parent.name
            if entry.get("baseline_trainer") == "isolated_generic":
                entry["checkpoint"] = str(
                    retrained_root / entry["id"] / f"k{args.shot}" / "model.pth"
                )
            else:
                entry["checkpoint"] = str(
                    retrained_root / entry["id"] / original_run_name / "model.pth"
                )
    scoped = _scope_entries(all_entries, args.scope, args.datasets)
    inventory = _preflight(scoped, args, registry)
    preflight_path = Path(args.output_root) / "summary" / f"preflight_k{args.shot}.json"
    _dump(preflight_path, inventory)
    print(json.dumps({
        "preflight": str(preflight_path),
        "datasets_in_scope": len(scoped),
        "warnings": inventory["warnings"],
        "errors": inventory["errors"],
    }, ensure_ascii=False, indent=2), flush=True)

    compatible = [
        entry for entry in scoped
        if entry["status"] == "ready"
        or (entry["status"] == "experimental_trainable" and args.include_experimental)
    ]
    hard_errors = list(inventory["errors"])
    if args.strict_inventory and any(entry["status"] == "incompatible" for entry in scoped):
        hard_errors.append("incompatible dataset present under --strict-inventory")
    if hard_errors and args.stage != "preflight":
        print("Preflight failed; inspect the JSON report.", file=sys.stderr)
        return 2
    if args.stage == "preflight":
        return 2 if hard_errors else 0

    gpus = _gpu_ids(args.gpus)
    defaults = registry["defaults"]
    job_results = []
    stages = (
        ["baseline", "propose", "evaluate", "class-specific", "export", "summary", "validate"]
        if args.stage == "all" else [args.stage]
    )
    for stage in stages:
        if stage == "baseline":
            jobs = _baseline_jobs(compatible, args, defaults)
            result = _run_jobs(
                jobs, gpus, args.baseline_workers_per_gpu, stage, args.dry_run
            )
        elif stage == "propose":
            jobs = _proposal_jobs(compatible, args, defaults)
            result = _run_jobs(
                jobs, gpus, args.proposal_workers_per_gpu, stage, args.dry_run
            )
        elif stage == "evaluate":
            jobs = _evaluation_jobs(compatible, args, defaults)
            result = _run_jobs(
                jobs, gpus, args.eval_workers_per_gpu, stage, args.dry_run
            )
        elif stage == "class-specific":
            jobs = _class_specific_jobs(compatible, args, defaults)
            result = _run_jobs(
                jobs, gpus, args.class_specific_workers_per_gpu, stage, args.dry_run
            )
        elif stage == "export":
            jobs = [_export_job(compatible, args)]
            result = _run_jobs(jobs, [gpus[0]], 1, stage, args.dry_run)
        elif stage == "summary":
            if not args.dry_run:
                _summarize(compatible, args, inventory, job_results)
            result = []
        elif stage == "validate":
            summary_path = (
                Path(args.output_root) / "summary" / f"multidataset_k{args.shot}.json"
            )
            validation_path = (
                Path(args.output_root) / "summary" / f"validation_k{args.shot}.json"
            )
            jobs = [Job(
                name=f"validate.multidataset.k{args.shot}",
                command=[
                    args.python,
                    str(PROJECT / "scripts" / "validate_multidataset_v6.py"),
                    "--summary", str(summary_path),
                    "--output", str(validation_path),
                    "--bootstrap", "1000",
                ],
                log_path=Path(args.output_root) / "logs" / "validate.log",
            )]
            result = _run_jobs(jobs, [gpus[0]], 1, stage, args.dry_run)
        else:
            raise AssertionError(stage)
        # Two INP processes per 24 GB card normally improve utilization.  If a
        # particular category has unusually large images/maps and OOMs, retry
        # only those jobs with one process per GPU instead of losing the run.
        oom_names = {row["job"] for row in result if _is_cuda_oom(row)}
        if oom_names and stage in {"evaluate", "class-specific"} and not args.dry_run:
            print(
                f"[{stage}] retrying {len(oom_names)} CUDA-OOM jobs at one worker/GPU",
                flush=True,
            )
            retry_jobs = [
                Job(
                    name=job.name,
                    command=job.command,
                    log_path=job.log_path.with_name(job.log_path.stem + ".retry.log"),
                )
                for job in jobs if job.name in oom_names
            ]
            retry_rows = _run_jobs(retry_jobs, gpus, 1, f"{stage}-oom-retry", False)
            for row in retry_rows:
                row["retried_after_cuda_oom"] = True
            result = [row for row in result if row["job"] not in oom_names] + retry_rows
        job_results.extend(result)
        failed = [row for row in result if row["returncode"] != 0]
        _dump(Path(args.output_root) / "summary" / "job_results.json", job_results)
        if failed:
            print(json.dumps({"stage": stage, "failed": failed}, indent=2), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
