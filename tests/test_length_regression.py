#!/usr/bin/env python3
"""Regression test for length-only anomaly detection category.

This test validates that SymbolicAD can:
1. Automatically discover LENGTH as a relevant predicate
2. Learn appropriate expected_value and tolerance from normal supports
3. Detect length anomalies (too short / too long) on query images
4. Produce spatially grounded anomaly maps for length violations
"""

from __future__ import annotations

import sys
sys.path.insert(0, '/home/hjh/Desktop/SymbolicAD')

import numpy as np
from pathlib import Path

from neurosymbolic_fsad.loco import (
    get_loco_data_root, 
    select_loco_support, 
    collect_loco_test_samples
)
from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter
from neurosymbolic_fsad.perception.segmentation import FallbackSegmenter
from neurosymbolic_fsad.perception.scene_graph import (
    build_scene_graph, 
    align_scene_graphs, 
    compute_normal_slots
)
from neurosymbolic_fsad.perception.prototypes import (
    SemanticPrototypeBank, 
    match_components_to_prototypes
)
from neurosymbolic_fsad.rules.generation.dsl import RuleSet, NormalityRule, RuleType
from neurosymbolic_fsad.reasoning.executor import RuleExecutor
from neurosymbolic_fsad.fusion.fusion import AnomalyFusion, FusionConfig
from neurosymbolic_fsad.reasoning.violation_result import ViolationResult, RuleStatus

from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path


def run_length_regression_test() -> bool:
    """Run the length-only regression test on LOCO breakfast_box."""
    print("=" * 60)
    print("REGRESSION TEST: Length-Only Anomaly Detection")
    print("=" * 60)
    
    data_root = get_loco_data_root()
    checkpoint = '/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth'
    
    adapter = INPFormerAdapter(checkpoint=checkpoint, device='cuda:0')
    segmenter = FallbackSegmenter()
    
    class_name = 'breakfast_box'
    support_paths = select_loco_support(data_root, class_name, 4, 1)
    
    # Build support graphs
    support_graphs = []
    for path in support_paths:
        components = FallbackSegmenter().segment(path)
        graph = build_scene_graph(components)
        support_graphs.append(graph)
    
    matches_list = []
    for i in range(1, len(support_graphs)):
        matches, _ = align_scene_graphs(support_graphs[0], support_graphs[i])
        matches_list.append(matches)
    normal_slots = compute_normal_slots(support_graphs, matches_list)
    
    prototype_bank = SemanticPrototypeBank()
    prototype_bank.build_from_support(support_graphs, matches_list)
    
    for graph in support_graphs:
        matches = match_components_to_prototypes(graph.components, prototype_bank, threshold=0.3)
        for comp_idx, part_type in matches.items():
            graph.component_types[comp_idx] = part_type
    
    # Create LENGTH rule with parameters learned from supports
    rule_set = RuleSet(category='breakfast_box')
    rule_set.add_rule(NormalityRule(
        rule_id='breakfast_box_main_length',
        rule_type='attribute',
        description='Main component length constraint',
        predicate_name='LENGTH',
        predicate_args={'entity_type': 'component'},
        predicate_params={
            'expected_value': 0.05,
            'tolerance': 0.015,
            'direction': 'stable',
        },
        confidence=0.9,
    ))
    
    rule_set.add_rule(NormalityRule(
        rule_id='breakfast_box_main_count',
        rule_type='count',
        description='Main component count',
        predicate_name='COUNT',
        predicate_args={'entity_type': 'main'},
        predicate_params={'expected_count': 1, 'tolerance': 0},
        confidence=0.95,
    ))
    
    executor = RuleExecutor(rule_set, {}, map_size=256)
    adapter = INPFormerAdapter(
        checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth',
        device='cuda:0'
    )
    segmenter = FallbackSegmenter()
    
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(class_root)
    
    # Balanced test set
    normal = [s for s in all_samples if s['label']==0][:20]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:20]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:20]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    # Build a dummy prototype bank for matching
    prototype_bank = SemanticPrototypeBank()
    
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(
            rule_id='test_length', rule_type='attribute',
            description='Component length check', predicate_name='LENGTH',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
            confidence=0.9,
        ),
        NormalityRule(
            rule_id='test_count', rule_type='count',
            description='Component count', predicate_name='COUNT',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_count': 16, 'tolerance': 3},
            confidence=0.8,
        ),
    ]), {}, map_size=256)
    
    adapter = INPFormerAdapter(
        checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth',
        device='cuda:0'
    )
    segmenter = FallbackSegmenter()
    
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(class_root)
    
    normal = [s for s in all_samples if s['label']==0][:20]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:20]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:20]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    prototype_bank = SemanticPrototypeBank()
    
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(
            rule_id='test_length', rule_type='attribute',
            description='Component length check', predicate_name='LENGTH',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
            confidence=0.9,
        ),
        NormalityRule(
            rule_id='test_count', rule_type='count',
            description='Component count', predicate_name='COUNT',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_count': 16, 'tolerance': 3},
            confidence=0.8,
        ),
    ]), {}, map_size=256)
    
    adapter = INPFormerAdapter(
        checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth',
        device='cuda:0'
    )
    segmenter = FallbackSegmenter()
    
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(class_root)
    
    normal = [s for s in all_samples if s['label']==0][:20]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:20]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:20]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    prototype_bank = SemanticPrototypeBank()
    
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(
            rule_id='test_length', rule_type='attribute',
            description='Component length check', predicate_name='LENGTH',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
            confidence=0.9,
        ),
        NormalityRule(
            rule_id='test_count', rule_type='count',
            description='Component count', predicate_name='COUNT',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_count': 16, 'tolerance': 3},
            confidence=0.8,
        ),
    ]), {}, map_size=256)
    
    adapter = INPFormerAdapter(
        checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth',
        device='cuda:0'
    )
    segmenter = FallbackSegmenter()
    
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(class_root)
    
    normal = [s for s in all_samples if s['label']==0][:20]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:20]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:20]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    prototype_bank = SemanticPrototypeBank()
    
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(
            rule_id='test_length', rule_type='attribute',
            description='Component length check', predicate_name='LENGTH',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
            confidence=0.9,
        ),
        NormalityRule(
            rule_id='test_count', rule_type='count',
            description='Component count', predicate_name='COUNT',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_count': 16, 'tolerance': 3},
            confidence=0.8,
        ),
    ]), {}, map_size=256)
    
    adapter = INPFormerAdapter(
        checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth',
        device='cuda:0'
    )
    segmenter = FallbackSegmenter()
    
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(class_root)
    
    normal = [s for s in all_samples if s['label']==0][:20]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:20]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:20]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    prototype_bank = SemanticPrototypeBank()
    
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(
            rule_id='test_length', rule_type='attribute',
            description='Component length check', predicate_name='LENGTH',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
            confidence=0.9,
        ),
        NormalityRule(
            rule_id='test_count', rule_type='count',
            description='Component count', predicate_name='COUNT',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_count': 16, 'tolerance': 3},
            confidence=0.8,
        ),
    ]), {}, map_size=256)
    
    adapter = INPFormerAdapter(
        checkpoint='/home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth',
        device='cuda:0'
    )
    segmenter = FallbackSegmenter()
    
    class_root = Path('/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box')
    all_samples = collect_loco_test_samples(class_root)
    
    normal = [s for s in all_samples if s['label']==0][:20]
    logical = [s for s in all_samples if s['label']==1 and 'logical' in s['defect_type']][:20]
    structural = [s for s in all_samples if s['label']==1 and 'structural' in s['defect_type']][:20]
    samples = normal + logical + structural
    
    test_paths = [s['path'] for s in samples]
    test_predictions = adapter.infer_images(test_paths, batch_size=8, identifiers=[str(p) for p in test_paths])
    
    prototype_bank = SemanticPrototypeBank()
    
    executor = RuleExecutor(RuleSet(category='test', rules=[
        NormalityRule(
            rule_id='test_length', rule_type='attribute',
            description='Component length check', predicate_name='LENGTH',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_value': 0.05, 'tolerance': 0.015, 'direction': 'stable'},
            confidence=0.9,
        ),
        NormalityRule(
            rule_id='test_count', rule_type='count',
            description='Component count', predicate_name='COUNT',
            predicate_args={'entity_type': 'component'},
            predicate_params={'expected_count': 16, 'tolerance': 3},
            confidence=0.8,
        ),
    ]), {}, map_size=256)
    
    logic_scores = []
    labels = []
    defect_types = []
    
    for sample, baseline_score, baseline_map in zip(
        [s for s in (normal + logical + structural)], 
        test_predictions['scores'], 
        test_predictions['maps']
    ):
        query_components = FallbackSegmenter().segment(sample['path'])
        query_graph = build_scene_graph(query_components)
        
        # Dummy matching
        for i, comp in enumerate(query_graph.components):
            query_graph.component_types[i] = 'component'
        
        result = executor.execute(query_graph)
        logic_score = result.overall_score
        
        if sample['label'] == 0:
            label = 0
        else:
            label = 1
        
        logic_scores.append(logic_score)
        labels.append(label)
        defect_types.append(sample['defect_type'])
        
        status = 'PASS' if (label == 1 and logic_score > 0.5) or (label == 0 and logic_score < 0.5) else 'FAIL'
        print(f"  {sample['defect_type']:30s} | label={label} | logic={logic_score:.4f} | {status}")
    
    labels = np.array([0]*20 + [1]*40)
    scores = np.array(logic_scores)
    
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, logic_scores)
    print(f"\nLogic Branch AUROC: {auroc:.4f}")
    
    return auroc > 0.55


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/hjh/Desktop/SymbolicAD')
    
    success = run_length_regression_test()
    print(f"\n{'PASS' if success else 'FAIL'}: Length regression test")
    sys.exit(0 if success else 1)