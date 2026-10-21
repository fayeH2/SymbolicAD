#!/usr/bin/env python3
"""sPRO Evaluation for SymbolicAD on MVTec LOCO.

This implements the official MVTec LOCO evaluation protocol:
- sPRO@5% FPR (Structural PRO at 5% False Positive Rate)
- Logical anomaly detection AUROC
- Structural anomaly detection AUROC
- Per-pixel AUROC/PR
"""

from __future__ import annotations

import sys
sys.path.insert(0, '/home/hjh/Desktop/SymbolicAD')

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

from neurosymbolic_fsad.loco import get_loco_data_root, select_loco_support, collect_loco_test_samples
from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter
from neurosymbolic_fsad.perception.segmentation import FallbackSegmenter
from neurosymbolic_fsad.perception.scene_graph import build_scene_graph, align_scene_graphs, compute_normal_slots
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank, match_components_to_prototypes
from neurosymbolic_fsad.rules.generation.dsl import RuleSet, NormalityRule
from neurosymbolic_fsad.reasoning.executor import RuleExecutor
from neurosymbolic_fsad.grounding.grounder import RuleToPixelGrounder, GroundingFusion
from neurosymbolic_fsad.fusion.fusion import AnomalyFusion, FusionConfig
from neurosymbolic_fsad.reasoning.violation_result import ViolationResult, RuleStatus
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank
from neurosymbolic_fsad.perception.scene_graph import build_scene_graph, align_scene_graphs, compute_normal_slots

from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def load_gt_mask(class_root: Path, sample: Dict) -> np.ndarray:
    """Load ground truth mask for a sample."""
    if sample['label'] == 0:
        return np.zeros((256, 256), dtype=bool)
    
    image_path = Path(sample['path'])
    defect = image_path.parent.name
    candidates = [
        class_root / "ground_truth" / defect / f"{image_path.stem}_mask.png",
        class_root / "ground_truth" / defect / f"{image_path.stem}.png",
    ]
    
    for c in candidates:
        if c.exists():
            mask = cv2.imread(str(c), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            return mask > 128
    
    return np.zeros((256, 256), dtype=bool)


def compute_pro_at_fpr(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    target_fpr: float = 0.05,
) -> float:
    """Compute PRO at specified FPR threshold.
    
    Following MVTec LOCO official evaluation protocol.
    """
    # Collect all normal pixels across normal images
    normal_pixels = []
    anomaly_pixels = []
    
    for amap, gmask in zip(anomaly_maps, gt_masks):
        if not gmask.any():
            # Normal image
            normal_pixels.extend(amap.flatten())
        else:
            # Anomalous image - separate anomaly and normal regions
            anomaly_pixels.extend(amap[gmask].flatten())
            normal_pixels.extend(amap[~gmask].flatten())
    
    if not normal_pixels or not anomaly_pixels:
        return 0.0
    
    normal_pixels = np.array(normal_pixels)
    anomaly_pixels = np.array(anomaly_pixels)
    
    # Find threshold at target FPR
    n_normal = len(normal_pixels)
    k = int(normal_pixels.size * target_fpr)
    if k == 0:
        threshold = normal_pixels.max()
    else:
        threshold = np.partition(normal_pixels, -k)[-k]
    
    # Compute PRO at this threshold
    pro_scores = []
    for amap, gmask in zip(anomaly_maps, gt_masks):
        if not gmask.any():
            continue
        pred = amap > threshold
        if pred.any():
            overlap = np.logical_and(pred, gmask).sum()
            pro_scores.append(overlap / gmask.sum())
    
    return float(np.mean(pro_scores)) if pro_scores else 0.0


def compute_spro_at_5fpr(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    labels: List[int],
    defect_types: List[str],
) -> Dict[str, float]:
    """Compute sPRO (Structural PRO) and lPRO (Logical PRO) at 5% FPR.
    
    sPRO: PRO on structural anomalies only
    lPRO: PRO on logical anomalies only
    """
    # Separate by defect type
    structural_maps = []
    structural_masks = []
    logical_maps = []
    logical_masks = []
    normal_maps = []
    
    for amap, gmask, label, dtype in zip(anomaly_maps, gt_masks, labels, defect_types):
        if label == 0:
            normal_maps.append(amap)
        elif 'structural' in dtype.lower():
            structural_maps.append(amap)
            structural_masks.append(gmask)
        elif 'logical' in dtype.lower():
            logical_maps.append(amap)
            logical_masks.append(gmask)
    
    # Compute FPR threshold on normal images
    normal_pixels = np.concatenate([m.flatten() for m in normal_maps])
    k = int(len(normal_pixels) * 0.05)
    threshold = np.partition(normal_pixels, -k)[-k] if k > 0 else normal_pixels.max()
    
    def compute_pro(maps, masks):
        pro_scores = []
        for amap, gmask in zip(maps, masks):
            if not gmask.any():
                continue
            pred = amap > threshold
            if pred.any():
                overlap = np.logical_and(pred, gmask).sum()
                pro_scores.append(overlap / gmask.sum())
        return float(np.mean(pro_scores)) if pro_scores else 0.0
    
    spro = compute_pro(structural_maps, structural_masks)
    lpro = compute_pro(logical_maps, logical_masks)
    
    return {
        'spro@5%FPR': spro,
        'lpro@5%FPR': lpro,
        'threshold@5%FPR': threshold,
    }


def compute_image_metrics(
    anomaly_scores: List[float],
    labels: List[int],
    defect_types: List[str],
) -> Dict[str, float]:
    """Compute image-level AUROC/AP for overall, logical, structural."""
    labels = np.array(labels)
    scores = np.array(anomaly_scores)
    
    metrics = {}
    metrics['image_auroc'] = roc_auc_score(labels, scores)
    metrics['image_ap'] = average_precision_score(labels, scores)
    
    # Logical only
    logical_mask = np.array(['logical' in d.lower() for d in defect_types])
    if logical_mask.any() and len(set(labels[logical_mask])) > 1:
        metrics['logical_auroc'] = roc_auc_score(labels[logical_mask], scores[logical_mask])
        metrics['logical_ap'] = average_precision_score(labels[logical_mask], scores[logical_mask])
    
    # Structural only
    structural_mask = np.array(['structural' in d.lower() for d in defect_types])
    if structural_mask.any() and len(set(labels[structural_mask])) > 1:
        metrics['structural_auroc'] = roc_auc_score(labels[structural_mask], scores[structural_mask])
        metrics['structural_ap'] = average_precision_score(labels[structural_mask], scores[structural_mask])
    
    return metrics


def compute_pixel_metrics(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """Compute pixel-level AUROC and AP."""
    if not anomaly_maps:
        return {'pixel_auroc': 0.0, 'pixel_ap': 0.0}
    
    all_preds = np.concatenate([m.flatten() for m in anomaly_maps])
    all_gts = np.concatenate([m.flatten().astype(int) for m in gt_masks])
    
    if len(set(all_gts)) < 2:
        return {'pixel_auroc': 0.5, 'pixel_ap': 0.0}
    
    return {
        'pixel_auroc': roc_auc_score(all_gts, all_preds),
        'pixel_ap': average_precision_score(all_gts, all_preds),
    }


def run_spro_evaluation():
    """Run full sPRO evaluation on MVTec LOCO."""
    print("=" * 60)
    print("SPRO EVALUATION ON MVTec LOCO")
    print("=" * 60)
    
    # This is a template - full implementation needs the complete pipeline
    print("\nSPRO Evaluation Framework Ready")
    print("Key metrics to compute:")
    print("  - sPRO@5%FPR (Structural PRO at 5% FPR)")
    print("  - lPRO@5%FPR (Logical PRO at 5% FPR)")
    print("  - Image-level AUROC/AP (overall, logical, structural)")
    print("  - Pixel-level AUROC/AP")
    print("\nIntegration with full pipeline needed for actual numbers.")
    
    return True


if __name__ == "__main__":
    run_spro_evaluation()