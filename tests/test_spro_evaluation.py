"""验证 LOCO 饱和区域重叠和逐 map-family 定位评价接口。"""

import numpy as np

from symbolicAD.evaluation.spro import (
    LocoEvaluationSample, SaturationSpec, evaluate_map_families, evaluate_spro,
)


def _sample(sample_id, anomaly_type, gt, **maps):
    return LocoEvaluationSample(
        sample_id, "toy", anomaly_type, gt, maps
    )


def test_spro_uses_saturation_threshold_instead_of_full_region_area():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 5:15] = 255
    prediction = np.zeros_like(gt, dtype=np.float32)
    prediction[5:10, 5:15] = 1.0  # half the annotated region
    good = np.zeros_like(gt)
    samples = [
        _sample("good", "good", good, logic_total=np.zeros_like(prediction)),
        _sample("logical", "logical", gt, logic_total=prediction),
    ]
    table = {"toy": {255: SaturationSpec(relative_area=0.5)}}
    curve = evaluate_spro(
        samples, "logic_total", "logical", threshold_count=20,
        saturation_table=table,
    )
    assert curve.region_count == 1
    assert curve.saturated_overlaps.max() == 1.0


def test_map_family_evaluation_reports_rule_logic_and_fused_separately():
    zero = np.zeros((20, 20), dtype=np.uint8)
    logical_gt = zero.copy(); logical_gt[4:8, 4:8] = 255
    structural_gt = zero.copy(); structural_gt[12:16, 12:16] = 254
    logical_map = (logical_gt > 0).astype(np.float32)
    structural_map = (structural_gt > 0).astype(np.float32)
    samples = [
        _sample("good", "good", zero,
                **{"rule/length": zero.astype(float), "logic_total": zero.astype(float),
                   "fused": zero.astype(float)}),
        _sample("logical", "logical", logical_gt,
                **{"rule/length": logical_map, "logic_total": logical_map,
                   "fused": logical_map}),
        _sample("structural", "structural", structural_gt,
                **{"rule/length": zero.astype(float), "logic_total": zero.astype(float),
                   "fused": structural_map}),
    ]
    table = {"toy": {
        255: SaturationSpec(relative_area=1.0),
        254: SaturationSpec(relative_area=1.0),
    }}
    report = evaluate_map_families(
        samples, threshold_count=20, saturation_table=table
    )
    assert set(report) == {"rule/length", "logic_total", "fused"}
    assert report["logic_total"]["logical_spro_5fpr"] > 0.9
    # A constant map has chance-level tie interpolation, not useful
    # localization; it remains far below the correctly localized fused map.
    assert report["logic_total"]["structural_spro_5fpr"] < 0.1
    assert report["fused"]["structural_spro_5fpr"] > 0.9
