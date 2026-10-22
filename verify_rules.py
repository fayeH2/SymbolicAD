from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, '/home/hjh/Desktop/SymbolicAD')

from symbolicAD.rules.facts import GroundingDinoDetector, SceneFactBuilder, SceneFactConfig
from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.rules.rule_schema import RuleSpec
from symbolicAD.rules.search.symbolicdet_evolution import evaluate_rule_set, CounterfactualPair
from symbolicAD.rules.qwen_adapter import adapt_qwen_candidates

def build_contexts_for_paths(paths):
    det = GroundingDinoDetector(device="cuda:0")
    builder = SceneFactBuilder(det, SceneFactConfig())
    ctxs = []
    for p in paths:
        ctxs.append(builder.build(p))
    return ctxs
