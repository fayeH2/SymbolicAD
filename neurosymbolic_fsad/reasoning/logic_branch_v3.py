"""Logic Branch v3: Targeted rules for logical anomalies using discriminative features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from ..rules.generation.dsl import NormalityRule, RuleSet, RuleType, AttributeType
from ..perception.scene_graph import SceneGraph


@dataclass
class LogicRuleConfig:
    """Configuration for a logic rule with calibrated thresholds."""
    rule_id: str
    rule_type: str  # 'count', 'attribute', 'spatial'
    feature_key: str  # key in scene_stats
    expected_mean: float
    expected_std: float
    threshold_multiplier: float = 3.0  # sigma threshold
    direction: str = 'both'  # 'high', 'low', 'both'
    weight: float = 1.0
    description: str = ""


class LogicBranchV3:
    """Logic branch with calibrated statistical rules for logical anomalies."""
    
    def __init__(
        self,
        rule_configs: List[LogicRuleConfig],
        normal_slots: Dict[int, Dict],
        map_size: int = 256,
    ):
        self.rule_configs = rule_configs
        self.normal_slots = normal_slots
        self.map_size = map_size
    
    def evaluate(self, graph: SceneGraph) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate all rules, return (overall_score, rule_scores, spatial_maps)."""
        from neurosymbolic_fsad.reasoning.logic_branch_v2 import extract_scene_statistics
        scene_stats = extract_scene_statistics(graph)
        
        rule_scores = {}
        spatial_maps = {}
        violations = []
        
        for config in self.rule_configs:
            score, witness = self._eval_rule(config, scene_stats)
            rule_scores[config.rule_id] = 1.0 - score
            
            if score > 1e-6:
                violations.append((config.rule_id, score, config.weight))
                # Create witness map
                if witness := self._make_witness(config, scene_stats):
                    spatial_maps[config.rule_id] = witness
        
        # Weighted overall score
        if violations:
            weighted = sum(s * w for _, s, w in violations)
            total_w = sum(w for _, _, w in violations)
            overall = weighted / max(total_w, 1e-6)
        else:
            overall = 0.0
        
        return overall, rule_scores, spatial_maps
    
    def _eval_rule(self, config: LogicRuleConfig, scene_stats: Dict) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate a single rule using z-score thresholding."""
        actual = scene_stats.get(config.feature_key)
        if actual is None:
            return 0.0, None
        
        mean = config.expected_mean
        std = max(config.expected_std, 1e-6)
        threshold = config.threshold_multiplier * std
        
        diff = actual - mean
        abs_diff = abs(diff)
        
        if config.direction == 'high':
            violation = max(0.0, (diff - threshold) / max(std, 1e-6))
        elif config.direction == 'low':
            violation = max(0.0, (-diff - threshold) / max(std, 1e-6))
        else:  # both
            violation = max(0.0, (abs_diff - threshold) / max(std, 1e-6))
        
        violation = min(1.0, violation)
        
        # Witness for high violations
        witness = None
        if violation > 0.3:
            witness = self._make_witness(config, actual, mean, std)
        
        return min(1.0, violation), witness
    
    def _make_witness(self, config: LogicRuleConfig, actual: float, mean: float, std: float) -> np.ndarray:
        """Create a simple witness map (placeholder - would need actual component locations)."""
        # For now, return a simple heatmap indicating violation
        # In practice, this would highlight specific components
        witness = np.zeros((256, 256), dtype=np.float32)
        # Could add indicator at center or corners based on violation type
        return witness


def create_logical_anomaly_rules(normal_stats: Dict) -> List:
    """Create rules specifically targeting logical anomalies."""
    from neurosymbolic_fsad.rules.generation.dsl import LogicRuleConfig
    
    rules = []
    
    # === KEY DISCRIMINATORS FOR LOGICAL ANOMALIES ===
    
    # 1. LR Balance - Logical anomalies often have extreme left/right imbalance
    if 'lr_balance' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='lr_balance_extreme',
            rule_type='attribute',
            feature_key='lr_balance',
            expected_mean=normal_stats['lr_balance']['mean'],
            expected_std=normal_stats['lr_balance']['std'],
            threshold_multiplier=2.5,  # 2.5 sigma
            direction='high',  # Logical anomalies have HIGH lr_balance (extreme imbalance)
            weight=2.0,
            description='Extreme left-right component imbalance (logical anomaly indicator)',
        ))
    
    # 2. Component count - Logical anomalies can have very few or very many components
    if 'num_components' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='num_components_low',
            rule_type='count',
            feature_key='num_components',
            expected_mean=normal_stats['num_components']['mean'],
            expected_std=normal_stats['num_components']['std'],
            threshold_multiplier=2.5,
            direction='low',  # Missing components
            weight=1.5,
            description='Too few components (missing items in logical anomaly)',
        ))
        rules.append(LogicRuleConfig(
            rule_id='num_components_high',
            rule_type='count',
            feature_key='num_components',
            expected_mean=normal_stats['num_components']['mean'],
            expected_std=normal_stats['num_components']['std'],
            threshold_multiplier=2.5,
            direction='high',  # Extra components
            weight=1.5,
            description='Too many components (extra items in logical anomaly)',
        ))
    
    # 3. Vertical spread - Logical anomalies may have unusual vertical distribution
    if 'vertical_spread' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='vertical_spread_high',
            rule_type='attribute',
            feature_key='vertical_spread',
            expected_mean=normal_stats['vertical_spread']['mean'],
            expected_std=normal_stats['vertical_spread']['std'],
            threshold_multiplier=2.5,
            direction='high',
            weight=1.0,
            description='Abnormal vertical component distribution',
        ))
    
    # 3. Main object position stability
    if 'main_obj_cx' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='main_obj_cx_shift',
            rule_type='attribute',
            feature_key='main_obj_cx',
            expected_mean=normal_stats['main_obj_cx']['mean'],
            expected_std=normal_stats['main_obj_cx']['std'],
            threshold_multiplier=3.0,
            direction='both',
            weight=0.8,
            description='Main object horizontal position shift',
        ))
    
    if 'main_obj_cy' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='main_obj_cy_shift',
            rule_type='attribute',
            feature_key='main_obj_cy',
            expected_mean=normal_stats['main_obj_cy']['mean'],
            expected_std=normal_stats['main_obj_cy']['std'],
            threshold_multiplier=3.0,
            direction='both',
            weight=0.8,
            description='Main object vertical position shift',
        ))
    
    # 4. Main object size stability
    if 'main_obj_area' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='main_obj_size_change',
            rule_type='attribute',
            feature_key='main_obj_area',
            expected_mean=normal_stats['main_obj_area']['mean'],
            expected_std=normal_stats['main_obj_area']['std'],
            threshold_multiplier=3.0,
            direction='both',
            weight=0.8,
            description='Main object size change',
        ))
    
    # 5. LR count balance (left vs right half counts)
    if 'left_half_count' in normal_stats and 'right_half_count' in normal_stats:
        rules.append(LogicRuleConfig(
            rule_id='left_right_count_diff',
            rule_type='attribute',
            feature_key='lr_balance',  # Reuse lr_balance
            expected_mean=normal_stats['lr_balance']['mean'],
            expected_std=normal_stats['lr_balance']['std'],
            threshold_multiplier=2.0,
            direction='high',
            weight=1.5,
            description='Large difference between left/right component counts',
        ))
    
    return [
        config for config in rules 
        if config.feature_key in ['lr_balance', 'num_components', 'vertical_spread', 
                                  'main_obj_cx', 'main_obj_cy', 'main_obj_area']
    ]


def build_logic_branch_v3(normal_stats: Dict, normal_slots: Dict, map_size: int = 256):
    """Factory to create configured LogicBranchV3."""
    rule_configs = create_logical_anomaly_rules(normal_stats)
    
    class SimpleLogicBranch:
        def __init__(self, configs, slots, map_size):
            self.configs = configs
            self.normal_slots = slots
            self.map_size = map_size
        
        def evaluate(self, graph):
            from neurosymbolic_fsad.reasoning.logic_branch_v2 import extract_scene_statistics
            scene_stats = extract_scene_statistics(graph)
            
            scores = {}
            violations = []
            spatial_maps = {}
            
            for config in self.configs:
                actual = scene_stats.get(config.feature_key)
                if actual is None:
                    scores[config.rule_id] = 1.0
                    continue
                
                mean = config.expected_mean
                std = max(config.expected_std, 1e-6)
                threshold = config.threshold_multiplier * std
                
                diff = actual - mean
                if config.direction == 'high':
                    viol = max(0.0, (diff - config.threshold_multiplier * std) / max(std, 1e-6))
                elif config.direction == 'low':
                    viol = max(0.0, (-diff - config.threshold_multiplier * std) / max(std, 1e-6))
                else:
                    viol = max(0.0, (abs(diff) - config.threshold_multiplier * std) / max(std, 1e-6))
                
                viol = min(1.0, viol)
                scores[config.rule_id] = 1.0 - viol
                
                if viol > 1e-6:
                    violations.append((config.rule_id, viol, config.weight))
            
            if violations:
                weighted = sum(v * w for _, v, w in violations)
                total_w = sum(w for _, _, w in violations)
                overall = weighted / max(total_w, 1e-6)
            else:
                overall = 0.0
            
            return overall, scores, {}
    
    return SimpleLogicBranch(create_logical_anomaly_rules(normal_stats), {}, 256)


def create_scene_rules(normal_stats: Dict) -> List:
    """Backward compatible interface."""
    return create_logical_anomaly_rules(normal_stats)