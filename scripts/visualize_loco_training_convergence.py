#!/usr/bin/env python3
"""Plot finite-loss convergence for one or more INP-Former training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history", action="append", required=True,
        help="LABEL=/absolute/or/relative/training_history.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = []
    for entry in args.history:
        label, separator, raw_path = entry.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--history must be LABEL=PATH")
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        runs.append((label, payload))

    # Do not silently drop a fifth LOCO class: ``zip(four_colours, runs)``
    # previously rendered only four of five histories.  Tab10 is stable and
    # supports every class in the declared benchmark table.
    colour_map = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for run_index, (label, payload) in enumerate(runs):
        colour = colour_map(run_index % colour_map.N)
        history = payload["history"]
        epochs = np.asarray([row["epoch"] for row in history])
        total = np.asarray([row["total_loss"] for row in history])
        reconstruction = np.asarray([row["reconstruction_loss"] for row in history])
        axes[0].plot(epochs, total, color=colour, linewidth=1.8, label=label)
        axes[1].plot(epochs, reconstruction, color=colour, linewidth=1.8, label=label)
        convergence = payload["convergence"]
        axes[0].annotate(
            f"{label}: last/first={convergence['last_over_first']:.3f}",
            xy=(epochs[-1], total[-1]), xytext=(-8, 10 + 15 * run_index),
            textcoords="offset points", ha="right", color=colour, fontsize=9,
        )
    for axis, title, ylabel in zip(
        axes,
        ("Total training loss", "Reconstruction loss"),
        ("Loss", "Cosine reconstruction loss"),
    ):
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        axis.grid(color="#D7DEE3", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("INP-Former convergence on leakage-free K=4 supports", weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, facecolor="white")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
