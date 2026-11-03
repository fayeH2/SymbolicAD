"""Qwen thinking text must not corrupt the final typed rule array."""

from symbolicAD.rules.generation.dsl import (
    parse_qwen_rules,
    parse_qwen_rules_strict,
)


def test_parser_uses_valid_final_array_after_bracketed_reasoning():
    response = '''Thinking about [counts, positions] first.
    ```json
    [{"rule_id":"r1","rule_type":"cardinality-ish",
      "description":"one part", "predicate_name":"COUNT",
      "predicate_args":{"entity_type":"part"},
      "predicate_params":{"expected_count":1,"tolerance":0},
      "confidence":0.9}]
    ```'''
    rules = parse_qwen_rules(response, "demo")
    assert len(rules) == 1
    assert rules[0].rule_id == "r1"
    assert rules[0].predicate_name == "COUNT"
    assert rules[0].rule_type.value == "custom"


def test_parser_accepts_wrapped_rules_object():
    rules = parse_qwen_rules(
        '{"rules":[{"rule_id":"r2","description":"length",'
        '"predicate_name":"LENGTH","predicate_args":{"entity_type":"wire"},'
        '"predicate_params":{"expected_value":1.0,"tolerance":0.1}}]}',
        "demo",
    )
    assert [rule.rule_id for rule in rules] == ["r2"]


def test_parser_ignores_longer_numeric_reasoning_arrays():
    response = (
        "Reasoning candidates [1, 2, 3, 4, 5]. Final: "
        '[{"rule_id":"r3","description":"one",'
        '"predicate_name":"COUNT","predicate_args":{"entity_type":"part"},'
        '"predicate_params":{"expected_count":1,"tolerance":0}}]'
    )
    rules = parse_qwen_rules(response, "demo")
    assert [rule.rule_id for rule in rules] == ["r3"]


def test_permissive_parser_ignores_longer_nonrule_object_array():
    response = (
        'Reasoning [{"step":1},{"step":2}]. Final: '
        '[{"rule_id":"r4","description":"one",'
        '"predicate_name":"COUNT","predicate_args":{"entity_type":"part"},'
        '"predicate_params":{"expected_count":1,"tolerance":0}}]'
    )
    assert [rule.rule_id for rule in parse_qwen_rules(response, "demo")] == ["r4"]


def test_strict_parser_rejects_trailing_prose_nan_and_duplicate_ids():
    valid = (
        '[{"rule_id":"r","description":"one","predicate_name":"COUNT",'
        '"predicate_args":{"entity_type":"part"},'
        '"predicate_params":{"expected_count":1,"tolerance":0},'
        '"confidence":0.9}]'
    )
    assert len(parse_qwen_rules_strict(valid, "demo")) == 1
    assert parse_qwen_rules_strict(valid + " trailing", "demo") == []
    assert parse_qwen_rules_strict(valid.replace("0.9", "NaN"), "demo") == []
    duplicated = "[" + valid[1:-1] + "," + valid[1:-1] + "]"
    assert parse_qwen_rules_strict(duplicated, "demo") == []
