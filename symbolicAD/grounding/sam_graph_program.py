"""Support-only SAM anchor matching and frozen geometric predicate execution.

Anchors are repeatable regions, not guaranteed semantic object instances.
Qwen may select sparse relations between these observed anchors. Numeric bounds
always come from supports. Query matching permits unmatched regions.
"""
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def describe(masks, patches):
    vectors, geometry = [], []
    h, w = patches.shape[1:]
    for mask in masks:
        weights = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        vector = (patches * weights[None]).sum(axis=(1, 2)) / max(weights.sum(), 1e-8)
        vectors.append(vector / max(np.linalg.norm(vector), 1e-8))
        ys, xs = np.nonzero(mask)
        geometry.append([mask.mean(), (xs.mean() + .5) / mask.shape[1],
                         (ys.mean() + .5) / mask.shape[0]])
    return (np.asarray(vectors, np.float32).reshape(-1, patches.shape[0]),
            np.asarray(geometry, np.float32).reshape(-1, 3))


def assignment(reference, query, threshold=.15):
    """Visual assignment with explicit dummy columns; no forced bad matches."""
    if not len(reference) or not len(query):
        return {}
    distance = np.clip(1 - reference @ query.T, 0, 2)
    augmented = np.concatenate([distance, np.full((len(reference), len(reference)), threshold)], axis=1)
    rows, cols = linear_sum_assignment(augmented)
    return {int(r): int(c) for r, c in zip(rows, cols)
            if c < len(query) and distance[r, c] < threshold}


def compile_program(vectors, geometry, masks):
    if len(vectors) != 2:
        raise ValueError('Initial SAM graph protocol requires exactly two normal supports')
    match = assignment(vectors[0], vectors[1])
    anchors, expected = [], []
    for left, right in sorted(match.items()):
        values = np.stack([geometry[0][left], geometry[1][right]])
        # Only repeatable anchors are admitted. No query label is inspected.
        if abs(np.log(max(values[0, 0], 1e-8) / max(values[1, 0], 1e-8))) > .7:
            continue
        if np.linalg.norm(values[0, 1:] - values[1, 1:]) > .25:
            continue
        prototype = vectors[0][left] + vectors[1][right]
        prototype /= max(np.linalg.norm(prototype), 1e-8)
        anchors.append({'id': len(anchors), 'support_ids': [left, right],
                        'prototype': prototype.tolist(), 'values': values.tolist(),
                        'match_threshold': .15})
        expected.append(np.maximum(masks[0][left], masks[1][right]).astype(np.float32))
    relations = []
    pairs = list(combinations(range(len(anchors)), 2))
    # Limit prompt/expression growth before LLM selection. Prefer nearby pairs
    # using normal geometry only; deterministic, category-independent budget.
    pairs.sort(key=lambda pair: np.linalg.norm(
        np.mean(np.asarray(anchors[pair[0]]['values'])[:, 1:], axis=0)
        - np.mean(np.asarray(anchors[pair[1]]['values'])[:, 1:], axis=0)))
    for a, b in pairs[:12]:
        first, second = np.asarray(anchors[a]['values']), np.asarray(anchors[b]['values'])
        delta = second[:, 1:] - first[:, 1:]
        for index, kind in enumerate(('dx', 'dy')):
            values = delta[:, index]
            if np.ptp(values) > .12:
                continue
            relations.append({'a': a, 'b': b, 'kind': kind,
                              'lower': float(values.min() - .06),
                              'upper': float(values.max() + .06), 'scale': .06})
    return {'version': 'sam_anchor_v1', 'anchors': anchors, 'relations': relations,
            'source': 'support_geometry_candidates', 'query_fitting': False}, np.asarray(expected)


def execute(program, expected, query_vectors, geometry, masks, shape):
    anchors = program['anchors']
    heat = np.zeros(shape, np.float32)
    if not anchors:
        return heat, 0., []
    reference = np.asarray([a['prototype'] for a in anchors], np.float32)
    matches = assignment(reference, query_vectors)
    traces = []
    matched_indices = set(matches.values())
    for query_index in range(len(query_vectors)):
        if query_index in matched_indices:
            continue
        mask = masks[query_index]
        heat = np.maximum(heat, mask.astype(np.float32))
        traces.append({'kind': 'unmatched_instance', 'query_mask': query_index,
                       'severity': 1.0})
    for i, anchor in enumerate(anchors):
        if i not in matches:
            # Missing anchor is uncertain under segmentation failure. Keep its
            # evidence explicit and avoid claiming a calibrated probability.
            severity = 1.0
            heat = np.maximum(heat, severity * expected[i])
            traces.append({'kind': 'missing_anchor', 'anchor': i, 'severity': severity})
    for rule in program['relations']:
        a, b = rule['a'], rule['b']
        if a not in matches or b not in matches:
            continue  # existence evidence was already emitted above
        axis = 1 if rule['kind'] == 'dx' else 2
        value = float(geometry[matches[b], axis] - geometry[matches[a], axis])
        violation = max(rule['lower'] - value, value - rule['upper'], 0)
        severity = float(1 - np.exp(-violation / rule['scale']))
        if severity:
            region = np.maximum(masks[matches[a]], masks[matches[b]])
            heat = np.maximum(heat, severity * region)
            traces.append({**rule, 'value': value, 'severity': severity})
    return heat, float(heat.max()), traces
