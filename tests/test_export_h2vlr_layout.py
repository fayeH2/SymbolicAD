import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.export_h2vlr_layout import export_results


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_h2vlr_export_writes_metrics_visuals_and_checkpoint_links(tmp_path):
    data_root = tmp_path / "data"
    class_root = data_root / "widget"
    good = class_root / "test" / "good" / "000.png"
    bad = class_root / "test" / "broken" / "001.png"
    mask = class_root / "ground_truth" / "broken" / "001_mask.png"
    for path in (good, bad, mask):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 24, 3), 80, dtype=np.uint8)).save(good)
    Image.fromarray(np.full((20, 24, 3), 120, dtype=np.uint8)).save(bad)
    mask_array = np.zeros((20, 24), dtype=np.uint8)
    mask_array[6:14, 8:16] = 255
    Image.fromarray(mask_array).save(mask)

    output = tmp_path / "results"
    run = output / "generic" / "toy" / "widget" / "k4"
    rule_library = run / "normality_rules_v4_generic.json"
    _write_json(rule_library, {"rules": []})
    _write_json(run / "metrics_v4_generic.json", {
        "rule_library": str(rule_library),
        "baseline": {"i_auroc": 0.75, "i_aupr": 0.80},
        "baseline_plus_rules": {"i_auroc": 1.0, "i_aupr": 1.0},
        "delta_i_auroc": 0.25,
        "delta_i_aupr": 0.20,
        "selected_rule_count": 1,
        "fusion": {"enabled": True},
        "normal_trigger_rate": 0.0,
        "anomaly_trigger_rate": 1.0,
    })
    prediction = run / "predictions_v4_generic.csv"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    with prediction.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "path", "defect_type", "label", "baseline_score", "baseline_z",
            "relation_score", "decision_score", "top_rule",
        ])
        writer.writeheader()
        writer.writerow({
            "path": good, "defect_type": "good", "label": 0,
            "baseline_score": 0.1, "baseline_z": 0, "relation_score": 0,
            "decision_score": 0.1, "top_rule": "abstain",
        })
        writer.writerow({
            "path": bad, "defect_type": "broken", "label": 1,
            "baseline_score": 0.2, "baseline_z": 1, "relation_score": 1,
            "decision_score": 1.2, "top_rule": "count",
        })
    maps = np.zeros((2, 16, 16), dtype=np.float32)
    maps[1, 5:11, 5:11] = 1.0
    cache = output / "cache" / "toy" / "widget" / "k4" / "cache" / "test_inp_v4.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        paths=np.asarray([str(good.resolve()), str(bad.resolve())]),
        maps=maps,
        scores=np.asarray([0.1, 0.2], dtype=np.float32),
        metadata_json=np.asarray(json.dumps({
            "input_size": 24, "crop_size": 20, "map_size": 16,
        })),
    )
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"toy checkpoint")
    config = tmp_path / "config.json"
    _write_json(config, {
        "datasets": [{
            "id": "toy", "status": "ready", "data_root": str(data_root),
            "classes": ["widget"], "test_subdir": "test",
            "checkpoint": str(checkpoint),
        }],
    })

    report = export_results(argparse.Namespace(
        config=str(config), output_root=str(output), shot=4, support_seed=1,
        datasets=None, visual_size=40, max_visualizations_per_class=0,
        histogram_bins=256,
    ))
    destination = output / "toy" / "k_4"
    main_csv = destination / "csv" / "Seed_1-results.csv"
    assert main_csv.is_file()
    assert "toy-widget,100.00" in main_csv.read_text(encoding="utf-8")
    assert len(list((destination / "imgs").glob("*.jpg"))) == 6
    assert (destination / "checkpoint" / "CLS-Seed_1-widget-check_point.pth").is_symlink()
    row = report["datasets"][0]["classes"][0]
    assert row["p_roc"] is not None
    assert row["anomaly_masks_missing"] == 0
    assert row["visualizations"] == 2
