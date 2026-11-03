"""Protocol gates keep diagnostic rule experiments out of the strict model."""

from symbolicAD.evaluation.protocol_admission import assess_strict_kshot_protocol


def test_strict_k4_requires_shared_support_and_no_extra_normal_images():
    support = ("a", "b", "c", "d")
    result = assess_strict_kshot_protocol(
        rule_support_hashes=support,
        qwen_support_hash_sets=(support, support),
        qwen_audit_provenance_valid=True,
        baseline_support_hashes=support,
        baseline_artifact_valid=True,
        extra_normal_hashes=(),
        test_detection_after_rule_freeze=True,
    )
    assert result.strict_kshot_protocol_eligible
    assert result.reason_codes == ()


def test_extra_validation_and_mismatched_baseline_are_diagnostic_only():
    support = ("a", "b", "c", "d")
    result = assess_strict_kshot_protocol(
        rule_support_hashes=support,
        qwen_support_hash_sets=(support,),
        qwen_audit_provenance_valid=True,
        baseline_support_hashes=("w", "x", "y", "z"),
        baseline_artifact_valid=True,
        extra_normal_hashes=("a", "val_1", "val_2"),
        test_detection_after_rule_freeze=True,
    )
    assert result.rule_mechanism_protocol_valid
    assert not result.strict_kshot_protocol_eligible
    assert result.extra_normal_count == 2
    assert "BASELINE_SUPPORT_MISMATCH" in result.reason_codes
    assert "EXTRA_NORMAL_IMAGES_USED_FOR_CALIBRATION_OR_SEARCH" in result.reason_codes


def test_missing_qwen_provenance_fails_even_without_extra_images():
    result = assess_strict_kshot_protocol(
        rule_support_hashes=("a", "b", "c", "d"),
        qwen_support_hash_sets=(),
        qwen_audit_provenance_valid=False,
        baseline_support_hashes=("a", "b", "c", "d"),
        baseline_artifact_valid=True,
        extra_normal_hashes=(),
        test_detection_after_rule_freeze=True,
    )
    assert not result.rule_mechanism_protocol_valid
    assert not result.strict_kshot_protocol_eligible
    assert result.reason_codes == (
        "QWEN_SUPPORT_PROVENANCE_MISMATCH",
        "QWEN_AUDIT_PROVENANCE_INVALID",
    )
