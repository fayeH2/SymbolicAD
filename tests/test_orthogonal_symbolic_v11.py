import numpy as np

from neurosymbolic_fsad.generic_relations import GenericComponentGrounder
from neurosymbolic_fsad.orthogonal_symbolic import _symbolic_candidate_pool


def test_llm_seed_is_not_expanded_with_unproposed_constraints():
    proposal = {
        "primary_entity": {
            "id": "visible_parts", "expected_count": 3, "arrangement": "row",
        },
        "rules": [{"id": "visible_count", "predicate": "count_eq"}],
    }
    pool = _symbolic_candidate_pool(proposal)["rules"]
    by_predicate = {row["predicate"]: row for row in pool}
    assert set(by_predicate) == {"count_eq"}
    assert by_predicate["count_eq"]["candidate_origin"] == "llm_seed"


def test_missing_count_localizer_marks_only_the_unmatched_normal_slot():
    size = 64
    state = {
        "kind": "generic_component_v1",
        "expected_count": 3,
        "arrangement": "row",
        "image_size": size,
        "threshold": 24.0,
        "min_area": 0.001,
        "morph": 1,
        "support_match_rate": 1.0,
        "prototype": {
            "centroid": [[0.25, 0.5], [0.50, 0.5], [0.75, 0.5]],
            "area": [0.02, 0.02, 0.02],
            "hu": [[0.0] * 7] * 3,
            "orientation": [[1.0, 0.0]] * 3,
            "lab": [[0.0, 128.0, 128.0]] * 3,
        },
    }
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[27:37, 11:21] = 0
    image[27:37, 43:53] = 0
    facts, maps, diagnostics = GenericComponentGrounder(
        3, "row", size, state=state
    ).extract_spatial(image)
    count_map = maps["count_error"]
    assert np.isclose(facts["count_error"], 1.0 / 3.0)
    assert diagnostics["unmatched_slots"] == [1]
    assert count_map[32, 32] > 0.5
    assert count_map[32, 16] < 0.1
    assert count_map[32, 48] < 0.1
