"""Rule DSL definition for neural-symbolic anomaly detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class RuleType(Enum):
    """Types of normality rules."""
    COUNT = "count"
    ATTRIBUTE = "attribute"
    SPATIAL = "spatial"
    TOPOLOGICAL = "topological"
    COMPOSITION = "composition"
    CONDITIONAL = "conditional"


class SpatialRelationType(Enum):
    """Spatial relation types."""
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    FAR = "far"
    ALIGNED_HORIZONTAL = "aligned_horizontal"
    ALIGNED_VERTICAL = "aligned_vertical"
    SYMMETRIC = "symmetric"


class TopologicalRelationType(Enum):
    """Topological relation types."""
    CONNECTED = "connected"
    TOUCHING = "touching"
    CONTAINS = "contains"
    INSIDE = "inside"
    DISJOINT = "disjoint"
    OVERLAPS = "overlaps"


class AttributeType(Enum):
    """Attribute types for components."""
    COLOR_H = "color_h"
    COLOR_S = "color_s"
    COLOR_V = "color_v"
    SHAPE_ASPECT = "shape_aspect"
    SHAPE_CIRCULARITY = "shape_circularity"
    SHAPE_SOLIDITY = "shape_solidity"
    AREA = "area"
    POSITION_X = "position_x"
    POSITION_Y = "position_y"


@dataclass
class RuleCondition:
    """Condition for conditional rules."""
    attribute: str
    operator: str
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        if self.attribute not in context:
            return False
        lhs = context[self.attribute]
        
        if self.operator == "==":
            return lhs == self.value
        elif self.operator == "!=":
            return lhs != self.value
        elif self.operator == ">":
            return lhs > self.value
        elif self.operator == "<":
            return lhs < self.value
        elif self.operator == ">=":
            return lhs >= self.value
        elif self.operator == "<=":
            return lhs <= self.value
        elif self.operator == "in":
            return lhs in self.value
        return False


@dataclass
class NormalityRule:
    """A single normality rule."""
    rule_id: str
    rule_type: RuleType
    description: str
    part_type: Optional[str] = None
    target_part_type: Optional[str] = None
    
    # Count rule
    expected_count: Optional[int] = None
    count_tolerance: int = 0
    
    # Attribute rule
    attribute: Optional[AttributeType] = None
    expected_value: Optional[float] = None
    tolerance: Optional[float] = None
    direction: str = "stable"
    
    # Spatial rule
    spatial_relation: Optional[SpatialRelationType] = None
    spatial_threshold: float = 0.1
    
    # Topological rule
    topological_relation: Optional[TopologicalRelationType] = None
    
    # Composition rule
    required_part_types: List[str] = field(default_factory=list)
    forbidden_combinations: List[List[str]] = field(default_factory=list)
    
    # Conditional rule
    conditions: List[RuleCondition] = field(default_factory=list)
    then_rule: Optional["NormalityRule"] = None
    else_rule: Optional["NormalityRule"] = None
    
    # Metadata
    confidence: float = 1.0
    reliability: float = 1.0
    spatial_groundable: bool = True
    grounding_strategy: str = "default"
    source: str = "qwen"
    support_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "description": self.description,
            "part_type": self.part_type,
            "target_part_type": self.target_part_type,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "spatial_groundable": self.spatial_groundable,
            "grounding_strategy": self.grounding_strategy,
            "source": self.source,
            "support_evidence": self.support_evidence,
        }
        
        if self.rule_type == RuleType.COUNT:
            data.update({
                "expected_count": self.expected_count,
                "count_tolerance": self.count_tolerance,
            })
        elif self.rule_type == RuleType.ATTRIBUTE:
            data.update({
                "attribute": self.attribute.value if self.attribute else None,
                "expected_value": self.expected_value,
                "tolerance": self.tolerance,
                "direction": self.direction,
            })
        elif self.rule_type == RuleType.SPATIAL:
            data.update({
                "spatial_relation": self.spatial_relation.value if self.spatial_relation else None,
                "spatial_threshold": self.spatial_threshold,
            })
        elif self.rule_type == RuleType.TOPOLOGICAL:
            data.update({
                "topological_relation": self.topological_relation.value if self.topological_relation else None,
            })
        elif self.rule_type == RuleType.COMPOSITION:
            data.update({
                "required_part_types": self.required_part_types,
                "forbidden_combinations": self.forbidden_combinations,
            })
        elif self.rule_type == RuleType.CONDITIONAL:
            data.update({
                "conditions": [
                    {"attribute": c.attribute, "operator": c.operator, "value": c.value}
                    for c in self.conditions
                ],
                "then_rule": self.then_rule.to_dict() if self.then_rule else None,
                "else_rule": self.else_rule.to_dict() if self.else_rule else None,
            })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalityRule":
        """Deserialize from dictionary."""
        rule = cls(
            rule_id=data["rule_id"],
            rule_type=RuleType(data["rule_type"]),
            description=data["description"],
            part_type=data.get("part_type"),
            target_part_type=data.get("target_part_type"),
            confidence=data.get("confidence", 1.0),
            reliability=data.get("reliability", 1.0),
            spatial_groundable=data.get("spatial_groundable", True),
            grounding_strategy=data.get("grounding_strategy", "default"),
            source=data.get("source", "qwen"),
            support_evidence=data.get("support_evidence", []),
        )
        
        if rule.rule_type == RuleType.COUNT:
            rule.expected_count = data.get("expected_count")
            rule.count_tolerance = data.get("count_tolerance", 0)
        elif rule.rule_type == RuleType.ATTRIBUTE:
            rule.attribute = AttributeType(data["attribute"]) if data.get("attribute") else None
            rule.expected_value = data.get("expected_value")
            rule.tolerance = data.get("tolerance")
            rule.direction = data.get("direction", "stable")
        elif rule.rule_type == RuleType.SPATIAL:
            rule.spatial_relation = SpatialRelationType(data["spatial_relation"]) if data.get("spatial_relation") else None
            rule.spatial_threshold = data.get("spatial_threshold", 0.1)
        elif rule.rule_type == RuleType.TOPOLOGICAL:
            rule.topological_relation = TopologicalRelationType(data["topological_relation"]) if data.get("topological_relation") else None
        elif rule.rule_type == RuleType.COMPOSITION:
            rule.required_part_types = data.get("required_part_types", [])
            rule.forbidden_combinations = data.get("forbidden_combinations", [])
        elif rule.rule_type == RuleType.CONDITIONAL:
            rule.conditions = [
                RuleCondition(c["attribute"], c["operator"], c["value"])
                for c in data.get("conditions", [])
            ]
            if data.get("then_rule"):
                rule.then_rule = cls.from_dict(data["then_rule"])
            if data.get("else_rule"):
                rule.else_rule = cls.from_dict(data["else_rule"])
        
        return rule
    
    def get_grounding_strategy(self) -> str:
        """Get the spatial grounding strategy for this rule."""
        if not self.spatial_groundable:
            return "none"
        if self.grounding_strategy != "default":
            return self.grounding_strategy
        
        if self.rule_type == RuleType.COUNT:
            if self.expected_count is not None:
                return "missing_slots" if self._implies_missing() else "extra_objects"
            return "count_mismatch"
        elif self.rule_type == RuleType.ATTRIBUTE:
            return "attribute_map"
        elif self.rule_type == RuleType.SPATIAL:
            return "position_deviation"
        elif self.rule_type == RuleType.TOPOLOGICAL:
            return "connection_map"
        elif self.rule_type == RuleType.COMPOSITION:
            return "composition_map"
        elif self.rule_type == RuleType.CONDITIONAL:
            if self.then_rule:
                return self.then_rule.get_grounding_strategy()
        return "default"
    
    def _implies_missing(self) -> bool:
        return self.expected_count is not None and self.expected_count > 0


@dataclass
class RuleSet:
    """Collection of normality rules for a category."""
    category: str
    rules: List[NormalityRule] = field(default_factory=list)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_rule(self, rule: NormalityRule) -> None:
        self.rules.append(rule)
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[NormalityRule]:
        return [r for r in self.rules if r.rule_type == rule_type]
    
    def get_rules_for_part(self, part_type: str) -> List[NormalityRule]:
        return [r for r in self.rules if r.part_type is None or r.part_type == part_type]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "version": self.version,
            "metadata": self.metadata,
            "rules": [r.to_dict() for r in self.rules],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleSet":
        rule_set = cls(
            category=data["category"],
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )
        rule_set.rules = [NormalityRule.from_dict(r) for r in data.get("rules", [])]
        return rule_set
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "RuleSet":
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)


QWEN_RULE_GENERATION_PROMPT = """
You are an expert in industrial visual inspection. Analyze the provided normal reference images 
of product category "{category}" and induce a comprehensive set of normality rules.

The images show K={k_shot} normal samples. Your task is to identify invariant normal patterns 
that should hold for ALL normal instances of this category.

Generate rules covering these semantic dimensions:
1. COUNT: Number of components of each type (e.g., "3 leads", "4 pins", "2 holes")
2. ATTRIBUTE: Color, shape, size constraints for each component type
3. SPATIAL: Relative positions (left/right, above/below, alignment, symmetry)
4. TOPOLOGICAL: Connections, containment, touching relations
5. COMPOSITION: Valid combinations of components
6. CONDITIONAL: Different normal subtypes with different configurations

For each rule, provide:
- rule_id: unique identifier
- rule_type: one of [COUNT, ATTRIBUTE, SPATIAL, TOPOLOGICAL, COMPOSITION, CONDITIONAL]
- description: human-readable explanation
- part_type: target component type (use consistent naming)
- target_part_type: for binary relations
- parameters: rule-specific parameters (see schema)
- confidence: 0.0-1.0, how confident you are this is a universal normal pattern
- reasoning: why this rule is valid based on the support images

Output as valid JSON array of rule objects.
"""

QWEN_RULE_VALIDATION_PROMPT = """
You are given a candidate normality rule and a normal reference image. 
Determine if the rule holds for this image.

Rule: {rule_description}
Rule type: {rule_type}
Parameters: {parameters}

Analyze the image and respond with:
- holds: true/false
- evidence: visual evidence supporting or contradicting the rule
- confidence: 0.0-1.0
"""


def parse_qwen_rules(response: str, category: str) -> List[NormalityRule]:
    """Parse Qwen-VL response into NormalityRule objects."""
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            rules_data = json.loads(json_str)
        else:
            rules_data = json.loads(response)
    except json.JSONDecodeError:
        return []
    
    rules = []
    for i, data in enumerate(rules_data):
        try:
            rule = NormalityRule(
                rule_id=data.get("rule_id", f"{category}_rule_{i}"),
                rule_type=RuleType(data["rule_type"].lower()),
                description=data.get("description", ""),
                part_type=data.get("part_type"),
                target_part_type=data.get("target_part_type"),
                confidence=float(data.get("confidence", 0.8)),
                source="qwen",
            )
            
            params = data.get("parameters", {})
            if rule.rule_type == RuleType.COUNT:
                rule.expected_count = params.get("expected_count")
                rule.count_tolerance = params.get("tolerance", 0)
            elif rule.rule_type == RuleType.ATTRIBUTE:
                rule.attribute = AttributeType(params.get("attribute"))
                rule.expected_value = params.get("expected_value")
                rule.tolerance = params.get("tolerance")
                rule.direction = params.get("direction", "stable")
            elif rule.rule_type == RuleType.SPATIAL:
                rule.spatial_relation = SpatialRelationType(params.get("relation"))
                rule.spatial_threshold = params.get("threshold", 0.1)
            elif rule.rule_type == RuleType.TOPOLOGICAL:
                rule.topological_relation = TopologicalRelationType(params.get("relation"))
            elif rule.rule_type == RuleType.COMPOSITION:
                rule.required_part_types = params.get("required_types", [])
                rule.forbidden_combinations = params.get("forbidden", [])
            elif rule.rule_type == RuleType.CONDITIONAL:
                rule.conditions = [
                    RuleCondition(c["attribute"], c["operator"], c["value"])
                    for c in params.get("conditions", [])
                ]
            
            rules.append(rule)
        except (KeyError, ValueError) as e:
            print(f"Failed to parse rule {i}: {e}")
            continue
    
    return rules