"""唯一正式规则主链：schema -> registry -> validation -> optimization。

历史 DSL 仍可通过其具体模块读取旧产物，但不再从包入口导出，避免实验
脚本误把“声明过的谓词”当作“已有可执行器的谓词”。
"""

from .rule_schema import PredicateKind, RuleSetSpec, RuleSpec
from .rule_registry import RuleCapability, RuleRegistry, default_rule_registry
from .rule_validator import RuleValidation, RuleValidator, ValidationStatus
from .rule_compiler import (
    CompileDisposition, CompileRecord, CompiledRuleSet,
    InductionDecision, InductionResult, RuleCompiler, SupportRuleInducer,
)
from .rule_optimizer import (
    GreedyRuleSetOptimizer, RuleDependencyCycle, RuleDependencyGraph,
    RuleOptimizationReport, UnsatisfiedRuleDependency,
)

__all__ = [
    "PredicateKind", "RuleSetSpec", "RuleSpec", "RuleCapability",
    "RuleRegistry", "default_rule_registry", "RuleValidation", "RuleValidator",
    "ValidationStatus", "CompileDisposition", "CompileRecord",
    "CompiledRuleSet", "InductionDecision", "InductionResult", "RuleCompiler",
    "SupportRuleInducer", "GreedyRuleSetOptimizer", "RuleDependencyCycle",
    "RuleDependencyGraph", "RuleOptimizationReport", "UnsatisfiedRuleDependency",
]
