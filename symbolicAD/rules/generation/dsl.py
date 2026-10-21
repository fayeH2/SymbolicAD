"""Extensible Rule DSL with Predicate Registry for SymbolicAD.

This replaces the fixed-enum DSL with an open predicate registry system
that supports user-defined predicates (LENGTH, multi-entity relations, etc.).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from typing_extensions import Self

import numpy as np


# ============================================================
# PREDICATE REGISTRY - Core of extensibility
# ============================================================

class PredicateArity(Enum):
    UNARY = 1
    BINARY = 2
    NARY = 3


class PredicateCategory(Enum):
    COUNT = "count"
    ATTRIBUTE = "attribute"
    SPATIAL = "spatial"
    TOPOLOGICAL = "topological"
    COMPOSITION = "composition"
    CONDITIONAL = "conditional"
    CUSTOM = "custom"


@dataclass
class PredicateSpec:
    """Specification for a predicate in the registry."""
    name: str
    category: PredicateCategory
    arity: PredicateArity
    # Expected argument types for validation
    arg_types: List[str] = field(default_factory=list)
    # Default parameter names
    param_names: List[str] = field(default_factory=list)
    # Whether this predicate can produce spatial anomaly maps
    spatially_groundable: bool = True
    # Default grounding strategy
    default_grounding_strategy: str = "instance_mask"
    # Description for documentation/LLM
    description: str = ""
    # Validator function (returns bool)
    validator: Optional[Callable[..., bool]] = None
    # Executor function (returns violation_score in [0,1])
    executor: Optional[Callable[..., float]] = None
    # Grounding function (returns anomaly_map)
    grounder: Optional[Callable[..., np.ndarray]] = None


class PredicateRegistry:
    """Global registry for extensible predicates.
    
    Supports built-in predicates (COUNT, COLOR, etc.) and user-defined
    predicates (LENGTH, CUSTOM_RELATION, etc.) registered at runtime.
    """
    
    _instance: Optional['PredicateRegistry'] = None
    
    def __new__(cls) -> 'PredicateRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._predicates: Dict[str, PredicateSpec] = {}
            cls._instance._register_builtins()
        return cls._instance
    
    def _register_builtins(self) -> None:
        """Register built-in predicates."""
        # COUNT predicates
        self.register(PredicateSpec(
            name="COUNT",
            category=PredicateCategory.COUNT,
            arity=PredicateArity.UNARY,
            arg_types=["entity_type"],
            param_names=["expected_count", "tolerance"],
            spatially_groundable=True,
            default_grounding_strategy="count_mismatch",
            description="Component count constraint"
        ))
        
        self.register(PredicateSpec(
            name="COUNT_RELATIVE",
            category=PredicateCategory.COUNT,
            arity=PredicateArity.BINARY,
            arg_types=["entity_type", "entity_type"],
            param_names=["ratio", "tolerance"],
            spatially_groundable=True,
            default_grounding_strategy="count_ratio",
            description="Ratio between two component counts"
        ))
        
        # ATTRIBUTE predicates
        for attr in ["LENGTH", "WIDTH", "HEIGHT", "AREA", "ASPECT_RATIO", 
                     "COLOR_H", "COLOR_S", "COLOR_V", "CIRCULARITY", "SOLIDITY"]:
            self.register(PredicateSpec(
                name=attr,
                category=PredicateCategory.ATTRIBUTE,
                arity=PredicateArity.UNARY,
                arg_types=["entity_type"],
                param_names=["expected_value", "tolerance", "direction"],
                spatially_groundable=True,
                default_grounding_strategy="attribute_map",
                description=f"Component {attr.lower()} constraint"
            ))
        
        # SPATIAL predicates
        for rel in ["LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "FAR",
                    "ALIGNED_HORIZONTAL", "ALIGNED_VERTICAL", "SYMMETRIC_ABOUT"]:
            self.register(PredicateSpec(
                name=rel,
                category=PredicateCategory.SPATIAL,
                arity=PredicateArity.BINARY,
                arg_types=["entity_type", "entity_type"],
                param_names=["threshold"],
                spatially_groundable=True,
                default_grounding_strategy="position_deviation",
                description=f"Spatial relation: {rel.lower()}"
            ))
        
        # TOPOLOGICAL predicates
        for rel in ["CONNECTED", "TOUCHING", "CONTAINS", "INSIDE", "DISJOINT", "OVERLAPS"]:
            self.register(PredicateSpec(
                name=rel,
                category=PredicateCategory.TOPOLOGICAL,
                arity=PredicateArity.BINARY,
                arg_types=["entity_type", "entity_type"],
                param_names=[],
                spatially_groundable=True,
                default_grounding_strategy="connection_map",
                description=f"Topological relation: {rel.lower()}"
            ))
        
        # COMPOSITION predicates
        self.register(PredicateSpec(
            name="REQUIRES",
            category=PredicateCategory.COMPOSITION,
            arity=PredicateArity.NARY,
            arg_types=["entity_type"],
            param_names=["required_types"],
            spatially_groundable=True,
            default_grounding_strategy="composition_map",
            description="Required component types must be present"
        ))
        
        self.register(PredicateSpec(
            name="FORBIDS",
            category=PredicateCategory.COMPOSITION,
            arity=PredicateArity.NARY,
            arg_types=["entity_type"],
            param_names=["forbidden_combinations"],
            spatially_groundable=True,
            default_grounding_strategy="composition_map",
            description="Forbidden component combinations"
        ))
        
        # CONDITIONAL predicates
        self.register(PredicateSpec(
            name="IF_THEN_ELSE",
            category=PredicateCategory.CONDITIONAL,
            arity=PredicateArity.NARY,
            arg_types=["condition", "then_rule", "else_rule"],
            param_names=["conditions"],
            spatially_groundable=True,
            default_grounding_strategy="conditional",
            description="Conditional rule: if condition then rule else rule"
        ))
        
        # Custom predicate template
        self.register(PredicateSpec(
            name="CUSTOM",
            category=PredicateCategory.CUSTOM,
            arity=PredicateArity.NARY,
            arg_types=["entity_type"],
            param_names=["function_name", "params"],
            spatially_groundable=False,
            default_grounding_strategy="custom",
            description="User-defined custom predicate"
        ))
    
    def register(self, spec: PredicateSpec) -> None:
        """Register a new predicate."""
        self._predicates[spec.name.upper()] = spec
    
    def get(self, name: str) -> Optional[PredicateSpec]:
        """Get predicate spec by name (case-insensitive)."""
        return self._predicates.get(name.upper())
    
    def list_by_category(self, category: PredicateCategory) -> List[PredicateSpec]:
        return [p for p in self._predicates.values() if p.category == category]
    
    def list_all(self) -> List[PredicateSpec]:
        return list(self._predicates.values())
    
    def validate(self, predicate_name: str, args: Dict[str, Any]) -> bool:
        """Validate predicate arguments."""
        spec = self.get(predicate_name)
        if not spec:
            return False
        # Check required args
        for arg_type in spec.arg_types:
            if arg_type not in args:
                return False
        return True


# Global registry instance
REGISTRY = PredicateRegistry()


# ============================================================
# EXTENSIBLE RULE DSL
# ============================================================

class RuleType(Enum):
    COUNT = "count"
    ATTRIBUTE = "attribute"
    SPATIAL = "spatial"
    TOPOLOGICAL = "topological"
    COMPOSITION = "composition"
    CONDITIONAL = "conditional"
    CUSTOM = "custom"


@dataclass
class RuleCondition:
    """Condition for conditional rules."""
    attribute: str
    operator: str
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
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
    """A single normality rule with extensible predicate support."""
    rule_id: str
    rule_type: RuleType
    description: str
    
    # Predicate-based specification (replaces fixed enum fields)
    predicate_name: str = ""
    predicate_args: Dict[str, Any] = field(default_factory=dict)
    predicate_params: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy fields for backward compatibility (deprecated)
    part_type: Optional[str] = None
    target_part_type: Optional[str] = None
    expected_count: Optional[int] = None
    count_tolerance: int = 0
    attribute: Optional[str] = None
    expected_value: Optional[float] = None
    tolerance: Optional[float] = None
    direction: str = "stable"
    spatial_relation: Optional[str] = None
    spatial_threshold: float = 0.1
    topological_relation: Optional[str] = None
    required_part_types: List[str] = field(default_factory=list)
    forbidden_combinations: List[List[str]] = field(default_factory=list)
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
    
    def __post_init__(self):
        # Auto-populate predicate from legacy fields if not set
        if not self.predicate_name and self.rule_type != RuleType.CUSTOM:
            self._sync_legacy_to_predicate()
    
    def _sync_legacy_to_predicate(self) -> None:
        """Map legacy fields to predicate specification."""
        mapping = {
            RuleType.COUNT: "COUNT",
            RuleType.ATTRIBUTE: self.attribute.upper() if self.attribute else "ATTRIBUTE",
            RuleType.SPATIAL: self.spatial_relation.upper() if self.spatial_relation else "SPATIAL",
            RuleType.TOPOLOGICAL: self.topological_relation.upper() if self.topological_relation else "TOPOLOGICAL",
            RuleType.COMPOSITION: "REQUIRES" if self.required_part_types else "FORBIDS",
            RuleType.CONDITIONAL: "IF_THEN_ELSE",
        }
        self.predicate_name = mapping.get(self.rule_type, "CUSTOM")
        
        # Map legacy args
        if self.rule_type == RuleType.COUNT:
            self.predicate_args = {"entity_type": self.part_type or "component"}
            self.predicate_params = {"expected_count": self.expected_count, "tolerance": self.count_tolerance}
        elif self.rule_type == RuleType.ATTRIBUTE:
            self.predicate_args = {"entity_type": self.part_type or "component"}
            self.predicate_params = {"expected_value": self.expected_value, "tolerance": self.tolerance, "direction": self.direction}
        elif self.rule_type == RuleType.SPATIAL:
            self.predicate_args = {"entity_type": self.part_type, "target_entity_type": self.target_part_type}
            self.predicate_params = {"threshold": self.spatial_threshold}
        elif self.rule_type == RuleType.TOPOLOGICAL:
            self.predicate_args = {"entity_type": self.part_type, "target_entity_type": self.target_part_type}
        elif self.rule_type == RuleType.COMPOSITION:
            self.predicate_args = {"entity_types": self.required_part_types + [t for combo in self.forbidden_combinations for t in combo]}
            self.predicate_params = {"required_types": self.required_part_types, "forbidden": self.forbidden_combinations}
        elif self.rule_type == RuleType.CONDITIONAL:
            self.predicate_args = {"condition": self.conditions}
            self.predicate_params = {"then_rule": self.then_rule, "else_rule": self.else_rule}
    
    def get_spec(self) -> Optional['PredicateSpec']:
        """Get predicate spec from registry."""
        return REGISTRY.get(self.predicate_name)
    
    def validate(self) -> bool:
        """Validate rule against registry."""
        spec = self.get_spec()
        if not spec:
            return False
        return REGISTRY.validate(self.predicate_name, self.predicate_args)
    
    def get_grounding_strategy(self) -> str:
        if not self.spatial_groundable:
            return "none"
        if self.grounding_strategy != "default":
            return self.grounding_strategy
        spec = self.get_spec()
        return spec.default_grounding_strategy if spec else "instance_mask"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "description": self.description,
            "predicate_name": self.predicate_name,
            "predicate_args": self.predicate_args,
            "predicate_params": self.predicate_params,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "spatial_groundable": self.spatial_groundable,
            "grounding_strategy": self.grounding_strategy,
            "source": self.source,
            "support_evidence": self.support_evidence,
        }
        # Include legacy fields for backward compatibility
        data.update({
            "part_type": self.part_type,
            "target_part_type": self.target_part_type,
            "expected_count": self.expected_count,
            "count_tolerance": self.count_tolerance,
            "attribute": self.attribute,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "direction": self.direction,
            "spatial_relation": self.spatial_relation,
            "spatial_threshold": self.spatial_threshold,
            "topological_relation": self.topological_relation,
            "required_part_types": self.required_part_types,
            "forbidden_combinations": self.forbidden_combinations,
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
            predicate_name=data.get("predicate_name", ""),
            predicate_args=data.get("predicate_args", {}),
            predicate_params=data.get("predicate_params", {}),
            confidence=data.get("confidence", 1.0),
            reliability=data.get("reliability", 1.0),
            spatial_groundable=data.get("spatial_groundable", True),
            grounding_strategy=data.get("grounding_strategy", "default"),
            source=data.get("source", "qwen"),
            support_evidence=data.get("support_evidence", []),
        )
        
        # Backward compatibility: load legacy fields
        for field_name in ["part_type", "target_part_type", "expected_count", "count_tolerance",
                           "attribute", "expected_value", "tolerance", "direction",
                           "spatial_relation", "spatial_threshold", "topological_relation",
                           "required_part_types", "forbidden_combinations"]:
            if field_name in data:
                setattr(rule, field_name, data[field_name])
        
        # Conditions
        rule.conditions = [
            RuleCondition(c["attribute"], c["operator"], c["value"])
            for c in data.get("conditions", [])
        ]
        
        # Nested rules
        if data.get("then_rule"):
            rule.then_rule = cls.from_dict(data["then_rule"])
        if data.get("else_rule"):
            rule.else_rule = cls.from_dict(data["else_rule"])
        
        return rule
    
    def get_spec(self) -> Optional['PredicateSpec']:
        return REGISTRY.get(self.predicate_name)


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


# ============================================================
# QWEN-VL PROMPTS (updated for extensible predicates)
# ============================================================

QWEN_RULE_GENERATION_PROMPT = """
You are an expert in industrial visual inspection. Analyze the provided normal reference images 
of product category "{category}" and induce a comprehensive set of normality rules.

The images show K={k_shot} normal samples. Your task is to identify invariant normal patterns 
that should hold for ALL normal instances of this category.

Available predicate categories (use these names exactly):
- COUNT: Number of components of each type
- LENGTH, WIDTH, HEIGHT, AREA, ASPECT_RATIO, COLOR_H, COLOR_S, COLOR_V, CIRCULARITY, SOLIDITY: Component attributes
- LEFT_OF, RIGHT_OF, ABOVE, BELOW, NEAR, FAR, ALIGNED_HORIZONTAL, ALIGNED_VERTICAL, SYMMETRIC_ABOUT: Spatial relations
- CONNECTED, TOUCHING, CONTAINS, INSIDE, DISJOINT, OVERLAPS: Topological relations
- REQUIRES, FORBIDS: Composition constraints
- IF_THEN_ELSE: Conditional rules
- CUSTOM: User-defined predicate (provide function_name and params)

For each rule, provide:
- rule_id: unique identifier
- predicate_name: one of the above predicate names
- predicate_args: {{"entity_type": "...", "target_entity_type": "..."}} (as needed)
- predicate_params: {{"expected_value": ..., "tolerance": ..., "threshold": ...}} (as needed)
- description: human-readable explanation
- confidence: 0.0-1.0
- reasoning: why this rule is valid based on the support images

Output as valid JSON array of rule objects.
"""

QWEN_RULE_VALIDATION_PROMPT = """
You are given a candidate normality rule and a normal reference image. 
Determine if the rule holds for this image.

Rule: {rule_description}
Predicate: {predicate_name}
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
                rule_type=RuleType(data.get("rule_type", "custom").lower()),
                description=data.get("description", ""),
                predicate_name=data.get("predicate_name", ""),
                predicate_args=data.get("predicate_args", {}),
                predicate_params=data.get("predicate_params", {}),
                confidence=float(data.get("confidence", 0.8)),
                source="qwen",
            )
            
            # Map legacy fields if present
            params = data.get("parameters", {})
            if "expected_count" in params:
                rule.expected_count = params["expected_count"]
                rule.count_tolerance = params.get("tolerance", 0)
            if "expected_value" in params:
                rule.expected_value = params["expected_value"]
                rule.tolerance = params.get("tolerance")
                rule.direction = params.get("direction", "stable")
            if "threshold" in params:
                rule.spatial_threshold = params["threshold"]
            
            rules.append(rule)
        except (KeyError, ValueError) as e:
            print(f"Failed to parse rule {i}: {e}")
            continue
    
    return rules


__all__ = [
    "PredicateArity", "PredicateCategory", "PredicateSpec", "PredicateRegistry", "REGISTRY",
    "RuleType", "RuleCondition", "NormalityRule", "RuleSet",
    "RuleType", "PredicateCategory", "PredicateArity",
    "QWEN_RULE_GENERATION_PROMPT", "QWEN_RULE_VALIDATION_PROMPT",
    "parse_qwen_rules",
]