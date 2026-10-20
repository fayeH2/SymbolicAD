#!/usr/bin/env python3
"""Fail fast when the selected Conda runtime cannot access requested GPUs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--llm-model", required=True)
    args = parser.parse_args()

    checks = {}
    errors = []
    try:
        import cv2
        import numpy
        import sklearn
        import torch
        import transformers
        from PIL import Image  # noqa: F401

        checks.update({
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
            "sklearn": sklearn.__version__,
            "qwen3_vl_class_available": hasattr(
                transformers, "Qwen3VLForConditionalGeneration"
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        })
        if not checks["qwen3_vl_class_available"]:
            errors.append("transformers has no Qwen3VLForConditionalGeneration")
        if not torch.cuda.is_available():
            errors.append(
                "PyTorch cannot access CUDA. Run nvidia-smi and check /dev/nvidia* "
                "or the container/device mapping."
            )
        requested = [int(token.strip()) for token in args.gpus.split(",") if token.strip()]
        if requested and torch.cuda.device_count() <= max(requested):
            errors.append(
                f"requested physical GPUs {requested}, but PyTorch sees "
                f"{torch.cuda.device_count()} device(s)"
            )
        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 2**30, 2),
                })
        checks["devices"] = devices
    except Exception as exc:
        errors.append(f"runtime import failed: {type(exc).__name__}: {exc}")

    model = Path(args.llm_model).expanduser().resolve()
    checks["llm_model"] = str(model)
    checks["llm_model_exists"] = model.is_dir()
    if not model.is_dir():
        errors.append(f"local Qwen model is missing: {model}")
    checks["errors"] = errors
    checks["ready"] = not errors
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
