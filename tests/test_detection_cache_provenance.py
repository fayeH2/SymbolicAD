"""Detection facts must be invalidated when perception configuration changes."""

from symbolicAD.rules.facts.pipeline import detector_cache_fingerprint


def test_vocabulary_change_invalidates_detection_cache_identity():
    first = detector_cache_fingerprint({"part": ("part",)})
    same = detector_cache_fingerprint({"part": ("part",)})
    changed = detector_cache_fingerprint({"part": ("component",)})
    assert first == same
    assert first != changed
