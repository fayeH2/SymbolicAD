"""Rules package for SymbolicAD."""

from neurosymbolic_fsad.rules.generation.dsl import (
    NormalityRule, RuleSet, RuleType, RuleCondition,
    PredicateArity, PredicateCategory, PredicateSpec, PredicateRegistry, REGISTRY,
    QWEN_RULE_GENERATION_PROMPT, QWEN_RULE_VALIDATION_PROMPT,
    parse_qwen_rules,
)

__all__ = [
    "NormalityRule", "RuleSet", "RuleType", "RuleCondition",
    "PredicateArity", "PredicateCategory", "PredicateSpec", "PredicateRegistry", "REGISTRY",
    "QWEN_RULE_GENERATION_PROMPT", "QWEN_RULE_VALIDATION_PROMPT",
    "parse_qwen_rules",
]