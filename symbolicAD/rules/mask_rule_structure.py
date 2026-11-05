"""Executable predicate templates and FSAD-only parameter binding.

Qwen authors symbolic structure, not numbers.  A valid template explicitly
declares its atoms, writes every rule as a predicate expression, distinguishes
hard from soft rules, and leaves calibratable parameters as ``null``.  The
offline FSAD compiler then fills those placeholders and records their source.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, Mapping, Sequence, Tuple


SCHEMA_VERSION = "mask_rule_template_v2"
FORMULA_OPS = {"NORMAL_IF_ALL", "ANOMALY_IF_ANY"}
REQUIRED_ATOM_DEFINITIONS = {
    "DistToRef(m)": "mask_nearest_reference_distance",
    "N(comp)": "component_instance_count",
    "Distance(m_i,m_j)": "mask_centroid_euclidean_distance",
    "Direction(m_i,m_j)": "mask_pair_principal_direction",
}

_DIST_RE = re.compile(
    r"^DistToRef\(m\)\s*(<=|<)\s*([\u03b8Θ][A-Za-z0-9_\u0370-\u03ff]*)$"
)
_COUNT_RE = re.compile(
    r"^N\(([A-Za-z_][A-Za-z0-9_]*)\)\s*(>=|>|==|<=|<)\s*(\d+)$"
)
_PAIR_RE = re.compile(
    r"^(Distance|Direction)\(([A-Za-z_][A-Za-z0-9_]*),\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)\)\s*(<=|<|>=|>)\s*"
    r"([\u03b8Θ][A-Za-z0-9_\u0370-\u03ff]*)$"
)


def parse_predicate(predicate: str) -> Dict[str, object]:
    """Parse a whitelisted predicate into an unambiguous execution operator."""
    predicate = str(predicate).strip()
    match = _DIST_RE.fullmatch(predicate)
    if match:
        return {
            "symbol": "DistToRef",
            "args": ["m"],
            "comparison": match.group(1),
            "parameter": match.group(2),
        }
    match = _COUNT_RE.fullmatch(predicate)
    if match:
        return {
            "symbol": "N",
            "args": [match.group(1)],
            "comparison": match.group(2),
            "constant": int(match.group(3)),
        }
    match = _PAIR_RE.fullmatch(predicate)
    if match:
        return {
            "symbol": match.group(1),
            "args": [match.group(2), match.group(3)],
            "comparison": match.group(4),
            "parameter": match.group(5),
        }
    raise ValueError(f"unsupported predicate expression: {predicate!r}")


def _declared_signature(symbol: str) -> str:
    return {
        "DistToRef": "DistToRef(m)",
        "N": "N(comp)",
        "Distance": "Distance(m_i,m_j)",
        "Direction": "Direction(m_i,m_j)",
    }[symbol]


def validate_mask_rule_structure(
    payload: Mapping[str, object], valid_groups: Sequence[str]
) -> Tuple[Dict[str, object], Dict[str, object]]:
    valid_groups = set(valid_groups)
    declared_atoms = {
        str(key).strip(): str(value).strip()
        for key, value in dict(payload.get("symbol_atoms", {}) or {}).items()
    }
    accepted, rejected = [], []
    seen = set()
    for raw in list(payload.get("rules", [])):
        rule = dict(raw)
        rule_id = str(rule.get("id", "")).strip()
        rule_type = str(rule.get("type", "")).strip().lower()
        predicate = str(rule.get("predicate", "")).strip()
        semantics = str(rule.get("semantics", "")).strip()
        params = dict(rule.get("params", {}) or {})
        reason = ""
        parsed = None
        if not rule_id or rule_id in seen:
            reason = "missing_or_duplicate_id"
        elif rule_type not in {"hard", "soft"}:
            reason = "type_must_be_hard_or_soft"
        else:
            try:
                parsed = parse_predicate(predicate)
            except ValueError:
                reason = "unsupported_predicate"
        if not reason and parsed is not None:
            signature = _declared_signature(str(parsed["symbol"]))
            if signature not in declared_atoms:
                reason = "predicate_atom_not_declared"
            elif parsed["symbol"] == "DistToRef" and rule_type != "soft":
                reason = "DistToRef_must_be_soft"
            elif parsed["symbol"] == "N" and rule_type != "hard":
                reason = "N_must_be_hard"
            elif parsed["symbol"] in {"Distance", "Direction"} and rule_type != "soft":
                reason = "pair_geometry_must_be_soft"
            elif parsed["symbol"] == "Direction":
                reason = "Direction_requires_circular_interval_template"
            elif parsed["symbol"] == "N" and parsed["args"][0] not in valid_groups:
                reason = "unknown_or_unstable_component"
            elif parsed["symbol"] in {"Distance", "Direction"}:
                left, right = parsed["args"]
                if left not in valid_groups or right not in valid_groups or left == right:
                    reason = "invalid_relation_components"
        if not reason and parsed is not None:
            parameter = parsed.get("parameter")
            if parameter is None:
                if params:
                    reason = "hard_rule_must_not_have_calibration_params"
            elif set(params) != {parameter}:
                reason = "params_must_exactly_match_predicate_placeholder"
            elif params[parameter] is not None:
                reason = "llm_parameter_must_be_null"
        if not reason and not semantics:
            reason = "missing_semantics"
        if reason:
            rejected.append({"rule": rule, "reason": reason})
            continue
        seen.add(rule_id)
        accepted_rule = {
            "id": rule_id,
            "type": rule_type,
            "predicate": predicate,
            "params": params,
            "scope": copy.deepcopy(rule.get("scope")),
            "semantics": semantics,
            "compiled_atom": parsed,
        }
        accepted.append(accepted_rule)

    added = []
    if not any(rule["compiled_atom"]["symbol"] == "DistToRef" for rule in accepted):
        rule_id = "R_MASK_DEVIATION"
        suffix = 0
        while rule_id in seen:
            suffix += 1
            rule_id = f"R_MASK_DEVIATION_{suffix}"
        rule = {
            "id": rule_id,
            "type": "soft",
            "predicate": "DistToRef(m) < θ_d",
            "params": {"θ_d": None},
            "scope": "all_masks",
            "semantics": "Every mask should remain close to the normal reference library.",
            "compiled_atom": parse_predicate("DistToRef(m) < θ_d"),
        }
        accepted.insert(0, rule)
        declared_atoms.setdefault(
            "DistToRef(m)", "Distance from mask m to its nearest normal FSAD prototype"
        )
        added.append({"rule": rule, "reason": "mandatory_fsad_safety_rule"})

    formula = dict(payload.get("formula", {}) or {})
    formula_op = str(formula.get("op", "")).upper()
    if formula_op not in FORMULA_OPS:
        formula_op = "NORMAL_IF_ALL"
    requested_ids = list(formula.get("children", []))
    accepted_ids = {rule["id"] for rule in accepted}
    ordered = [rule_id for rule_id in requested_ids if rule_id in accepted_ids]
    ordered.extend(rule["id"] for rule in accepted if rule["id"] not in ordered)
    validated = {
        "schema_version": SCHEMA_VERSION,
        "category": str(payload.get("category", "")),
        "source": str(payload.get("source", "qwen_structure_only")),
        "symbol_atoms": declared_atoms,
        "rules": accepted,
        "formula": {
            "op": formula_op,
            "children": ordered,
            "note": str(formula.get("note", "All predicates describe normality; any violation is anomalous.")),
        },
    }
    report = {
        "rejected": rejected,
        "added": added,
        "undeclared_known_atoms": [
            key for key in REQUIRED_ATOM_DEFINITIONS if key not in declared_atoms
        ],
    }
    return validated, report


def bind_fsad_parameters(
    structure: Mapping[str, object],
    mask_threshold: float,
    group_thresholds: Mapping[str, float],
    relation_thresholds: Mapping[str, float],
    calibration_metadata: Mapping[str, object],
) -> Dict[str, object]:
    """Fill only null placeholders and attach a per-parameter provenance map."""
    program = copy.deepcopy(dict(structure))
    bound_rules = []
    for raw in program.get("rules", []):
        rule = copy.deepcopy(dict(raw))
        atom = dict(rule.get("compiled_atom") or parse_predicate(rule["predicate"]))
        symbol = atom["symbol"]
        bindings = {}
        parameter = atom.get("parameter")
        if symbol == "DistToRef":
            value = float(mask_threshold)
            parameter_key = "mask_distance.global_q95"
        elif symbol == "N":
            value = None
            parameter_key = None
        elif symbol in {"Distance", "Direction"}:
            left, right = atom["args"]
            key = f"{left}::{right}"
            reverse = f"{right}::{left}"
            selected = key if key in relation_thresholds else reverse
            value = float(relation_thresholds[selected])
            parameter_key = f"relation.{selected}.{symbol.lower()}_q95"
        else:  # pragma: no cover - parser whitelist
            raise ValueError(f"cannot bind symbol {symbol!r}")

        if parameter is not None:
            if rule.get("params", {}).get(parameter, "missing") is not None:
                raise ValueError(
                    f"rule {rule['id']} placeholder {parameter} was not null before binding"
                )
            rule["params"][parameter] = value
            bindings[parameter] = {
                "value": value,
                "parameter_key": parameter_key,
                "source": "normal_fsad_reference_only",
            }
        rule["compiled_atom"] = atom
        rule["bindings"] = bindings
        bound_rules.append(rule)
    program["rules"] = bound_rules
    program["calibration_metadata"] = dict(calibration_metadata)
    program["runtime_llm_required"] = False
    program["template_parameters_fully_bound"] = all(
        all(value is not None for value in rule.get("params", {}).values())
        for rule in bound_rules
    )
    return program
