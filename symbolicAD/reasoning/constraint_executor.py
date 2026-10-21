"""规则集执行与 Strong Kleene 三值正常性逻辑。

这里组合的是“正常性命题”：AND 中任一规则违反即可确认异常；OR 中必须所有
分支都违反才能确认异常。只要最终状态不是 ``VIOLATED``，总逻辑图就保持为零，
防止 UNKNOWN 的局部热图泄漏进最终结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

from .predicate_executor import ExecutionContext
from .violation_result import RuleStatus, ViolationResult

if TYPE_CHECKING:
    from ..rules.rule_compiler import CompiledRuleSet


class ConstraintOperator(str, Enum):
    AND = "AND"
    OR = "OR"


def combine_statuses(
    statuses: Iterable[RuleStatus], operator: ConstraintOperator = ConstraintOperator.AND
) -> RuleStatus:
    values = tuple(RuleStatus(value) for value in statuses)
    if not values:
        return RuleStatus.UNKNOWN
    # Strong Kleene truth table 被集中在一个函数中，便于回归测试穷举检查。
    if operator == ConstraintOperator.AND:
        if RuleStatus.VIOLATED in values:
            return RuleStatus.VIOLATED
        if all(value == RuleStatus.SATISFIED for value in values):
            return RuleStatus.SATISFIED
        return RuleStatus.UNKNOWN
    if RuleStatus.SATISFIED in values:
        return RuleStatus.SATISFIED
    if all(value == RuleStatus.VIOLATED for value in values):
        return RuleStatus.VIOLATED
    return RuleStatus.UNKNOWN


@dataclass(frozen=True)
class ConstraintExecution:
    status: RuleStatus
    violation_score: float
    evidence_confidence: float
    anomaly_map: np.ndarray
    rule_results: Tuple[ViolationResult, ...]
    satisfied_rule_ids: Tuple[str, ...]
    violated_rule_ids: Tuple[str, ...]
    unknown_rule_ids: Tuple[str, ...]


class ConstraintExecutor:
    """只执行编译账本中标记为 EXECUTABLE 的规则，并保留全部逐规则结果。"""
    def execute(
        self,
        compiled: CompiledRuleSet,
        context: ExecutionContext,
        *,
        operator: ConstraintOperator = ConstraintOperator.AND,
    ) -> ConstraintExecution:
        results = tuple(
            record.capability.executor.execute(record.rule, context)
            for record in compiled.executable_records
            if record.capability is not None
        )
        status = combine_statuses((row.status for row in results), operator)
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        # An OR normality expression is anomalous only when every executable
        # disjunct is known to be violated. UNKNOWN therefore cannot leak a map.
        contributing = (
            [row for row in results if row.status == RuleStatus.VIOLATED]
            if status == RuleStatus.VIOLATED else []
        )
        for row in contributing:
            if row.anomaly_map is not None:
                anomaly = np.maximum(anomaly, row.anomaly_map)
        score = max((row.violation_score for row in contributing), default=0.0)
        known = [row for row in results if row.status != RuleStatus.UNKNOWN]
        confidence = (
            float(np.mean([row.evidence_confidence for row in known]))
            if known else 0.0
        )
        return ConstraintExecution(
            status=status,
            violation_score=score,
            evidence_confidence=confidence,
            anomaly_map=anomaly,
            rule_results=results,
            satisfied_rule_ids=tuple(
                row.rule_id for row in results if row.status == RuleStatus.SATISFIED
            ),
            violated_rule_ids=tuple(
                row.rule_id for row in results if row.status == RuleStatus.VIOLATED
            ),
            unknown_rule_ids=tuple(
                row.rule_id for row in results if row.status == RuleStatus.UNKNOWN
            ),
        )


__all__ = [
    "ConstraintOperator", "ConstraintExecution", "ConstraintExecutor",
    "combine_statuses",
]
