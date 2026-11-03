"""Baseline / logic map normalization and score fusion."""

from __future__ import annotations

import numpy as np


def normalize_map(
    values: np.ndarray,
    q_low: float,
    q_high: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    denominator = max(float(q_high) - float(q_low), epsilon)
    return np.clip((np.asarray(values, dtype=np.float32) - float(q_low)) / denominator, 0.0, 1.0)


def topk_mean(maps: np.ndarray, percent: float = 1.0) -> np.ndarray:
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim == 1:
        maps = maps[None, :]
    flat = maps.reshape(maps.shape[0], -1)
    count = max(1, int(round(flat.shape[1] * float(percent) / 100.0)))
    count = min(count, flat.shape[1])
    partitioned = np.partition(flat, flat.shape[1] - count, axis=1)
    return partitioned[:, -count:].mean(axis=1).astype(np.float32)


def fuse_maps(
    base_map: np.ndarray,
    logic_map: np.ndarray,
    lambda_logic: float = 0.5,
    epsilon: float = 1e-6,
) -> np.ndarray:
    base_map = np.clip(np.asarray(base_map, dtype=np.float32), 0.0, 1.0)
    logic_map = np.clip(np.asarray(logic_map, dtype=np.float32), 0.0, 1.0)
    fused = 1.0 - (1.0 - base_map) * (1.0 - float(lambda_logic) * logic_map)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def logic_image_score(
    logic_map: np.ndarray,
    root_truth: float,
    topk_percent: float = 1.0,
) -> float:
    heat_score = float(topk_mean(logic_map[None, ...], topk_percent)[0])
    return float(max(1.0 - float(root_truth), heat_score))


def final_image_score(
    baseline_score: float,
    logic_score: float,
    beta: float = 0.5,
) -> float:
    beta = float(min(1.0, max(0.0, beta)))
    return float((1.0 - beta) * float(baseline_score) + beta * float(logic_score))
