#!/usr/bin/env python3
"""3-Tier Grounding Evaluation for SymbolicAD.

This test evaluates rule grounding at three levels:
1. Oracle: Human-confirmed rules + GT component masks (upper bound)
2. Perception: Human-confirmed rules + Auto perception (perception error)
3. End-to-end: Auto rules + Auto perception (full system)
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


def compute_pro_metric(anomaly_maps: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """Compute PRO (Per-Region Overlap) metric."""
    # Simplified PRO computation
    pro_scores = []
    for amap, gmask in zip(anomaly_maps, gt_masks):
        if not gmask.any():
            continue
        # Threshold anomaly map at various levels
        thresholds = np.linspace(0, 1, 20)
        pros = []
        for t in thresholds:
            pred = amap > t
            if pred.any():
                overlap = np.logical_and(pred, gmask).sum()
                pros.append(overlap / gmask.sum())
        if pros:
            pro_scores.append(np.mean(pros))
    return float(np.mean(pro_scores)) if pro_scores else 0.0


def run_3tier_grounding_test():
    """Run 3-tier grounding evaluation."""
    print("=" * 60)
    print("3-TIER GROUNDING EVALUATION")
    print("=" * 60)
    
    data_root = get_loco_data_root()
    checkpoint = '/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth'
    
    adapter = INPFormerAdapter(checkpoint=checkpoint, device='cuda:0')
    segmenter = FallbackSegmenter()
    
    class_name = 'breakfast_box'
    support_paths = select_loco_support(data_root, class_name, 4, 1)
    
    # Build support setup (shared across tiers)
    support_graphs = []
    for path in select_loco_support(data_root, class_name, 4, 1):
        comps = FallbackSegmenter().segment(path)
        support_graphs.append(build_scene_graph(comps))
    
    matches_list = []
    for i in range(1, len(support_graphs)):
        matches, _ = align_scene_graphs(support_graphs[0], support_graphs[i])
        matches_list.append(matches)
    normal_slots = compute_normal_slots(support_graphs, matches_list)
    
    prototype_bank = SemanticPrototypeBank()
    prototype_bank.build_from_support(support_graphs, matches_list)
    
    # Create test rules
    rule_set = RuleSet(category='breakfast_box')
    rule_set.add_rule(NormalityRule(
        rule_id='test_length', rule_type='attribute',
        description='Component length check', predicate_name='LENGTH',
        predicate_args={'entity_type': 'component'},
        predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
        confidence=0.9,
    ))
    rule_set.add_rule(NormalityRule(
        rule_id='test_count', rule_type='count',
        description='Component count', predicate_name='COUNT',
        predicate_args={'entity_type': 'component'},
        predicate_params={'expected_count': 16, 'tolerance': 3},
        confidence=0.8,
    ))
    
    # Load test data
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO') / 'breakfast_box')
    
    normal = [s for s in all_samples if s['label']==0][:10]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:10]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:10]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    gt_masks = [load_gt_mask(Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box'), s) for s in samples]
    labels = np.array([0]*10 + [1]*20)
    
    adapter = INPFormerAdapter(checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth', device='cuda:0')
    segmenter = FallbackSegmenter()
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    # Grounding components
    grounder = RuleToPixelGrounder(map_size=256)
    fusion = GroundingFusion(map_size=256)
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(rule_id='test_length', rule_type='attribute', description='Component length check', predicate_name='LENGTH',
                      predicate_args={'entity_type': 'component'},
                      predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'}, confidence=0.9),
        NormalityRule(rule_id='test_count', rule_type='count', description='Component count', predicate_name='COUNT',
                      predicate_args={'entity_type': 'component'},
                      predicate_params={'expected_count': 16, 'tolerance': 3}, confidence=0.8),
    ]), {}, map_size=256)
    
    prototype_bank = SemanticPrototypeBank()
    
    # Store results per tier
    tier_results = {}
    
    for tier_name in ['Oracle', 'Perception', 'End-to-End']:
        print(f"\n--- {tier_name} Tier ---")
        
        anomaly_maps = []
        logic_scores = []
        fused_scores = []
        visual_scores = []
        
        for sample, baseline_score, baseline_map in zip(samples, adapter.infer_images(test_paths, batch_size=8)['scores'], adapter.infer_images(test_paths, batch_size=8)['maps']):
            # This is a placeholder - in real implementation, each tier would use different rule/perception sources
            pass
        
        # Placeholder results
        tier_results[tier_name] = {
            'pixel_auroc': 0.0,
            'pixel_aupr': 0.0,
            'pro': 0.0,
        }
    
    print("\n3-Tier Grounding Evaluation Complete")
    print("Results (placeholder - full implementation needed):")
    for tier, metrics in tier_results.items():
        print(f"  {tier}: Pixel AUROC={metrics['pixel_auroc']:.4f}, PRO={metrics['pro']:.4f}")
    
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/hjh/Desktop/SymbolicAD')
    
    from neurosymbolic_fsad.loco import get_loco_data_root, select_loco_support, collect_loco_test_samples
    from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter
    from neurosymbolic_fsad.perception.segmentation import FallbackSegmenter
    from neurosymbolic_fsad.perception.scene_graph import build_scene_graph, align_scene_graphs, compute_normal_slots
    from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank
    from neurosymbolic_fsad.rules.generation.dsl import RuleSet, NormalityRule
    from neurosymbolic_fsad.reasoning.executor import RuleExecutor
    from neurosymbolic_fsad.grounding.grounder import RuleToPixelGrounder, GroundingFusion
    from neurosymbolic_fsad.fusion.fusion import AnomalyFusion, FusionConfig
    from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank
    from neurosymbolic_fsad.perception.scene_graph import build_scene_graph, align_scene_graphs, compute_normal_slots
    from neurosymbolic_fsad.rules.generation.dsl import RuleSet, NormalityRule
    from neurosymbolic_fsad.reasoning.executor import RuleExecutor
    from neurosymbolic_fsad.grounding.grounder import RuleToPixelGrounder, GroundingFusion
    from neurosymbolic_fsad.fusion.fusion import AnomalyFusion, FusionConfig
    from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank
    from neurosymbolic_fsad.perception.scene_graph import build_scene_graph, align_scene_graphs, compute_normal_slots
    from neurosymbolic_fsad.loco import collect_loco_test_samples
    from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter
    from neurosymbolic_fsad.perception.segmentation import FallbackSegmenter
    
    run_3tier_grounding_test()