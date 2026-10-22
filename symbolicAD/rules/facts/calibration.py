"""Perception completeness calibration for count / presence predicates.

The rule executors refuse to turn "the detector found nothing" into a logical
"the object is missing" unless the detector has been calibrated on independent
normal images.  This module produces exactly that calibration:

``count_observation_*``
    Exact-count reliability.  Requires both that the entity is seen at all and
    that its cardinality is stable, so an entity whose instance count flickers
    (touch-merged banana slices) never authorises an exact-count decision.

``presence_observation_*``
    Presence reliability.  Only requires that the entity is detected whenever
    it is actually present, which is the correct evidence for ``exists``.

Calibration is estimated on the *train* split, never on the validation split
used for ``L_normal``, and never on the query anomalies.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EntityCalibration:
    entity_type: str
    sample_count: int
    detection_rate: float
    count_stability: float
    modal_count: int
    mean_top_score: float
    count_min: int
    count_max: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "sample_count": self.sample_count,
            "detection_rate": self.detection_rate,
            "count_stability": self.count_stability,
            "modal_count": self.modal_count,
            "mean_top_score": self.mean_top_score,
            "count_min": self.count_min,
            "count_max": self.count_max,
        }


@dataclass(frozen=True)
class CountCalibration:
    """Calibration-derived observation completeness for one dataset split."""

    entities: Mapping[str, EntityCalibration]
    split: str
    source_images: Tuple[str, ...] = ()
    minimum_detection_rate: float = 0.9
    minimum_count_stability: float = 0.9
    minimum_presence_score: float = 0.3

    def _entity(self, entity_type: str) -> Optional[EntityCalibration]:
        return self.entities.get(entity_type)

    def count_complete(self, entity_type: str) -> bool:
        row = self._entity(entity_type)
        if row is None:
            return False
        return bool(
            row.detection_rate >= self.minimum_detection_rate
            and row.count_stability >= self.minimum_count_stability
            and row.mean_top_score >= self.minimum_presence_score
        )

    def count_confidence(self, entity_type: str) -> float:
        """Completeness estimate = how often the exact count was reproduced.

        The detector score is an admissibility gate, not a completeness
        probability, so it is deliberately excluded here.
        """
        row = self._entity(entity_type)
        if row is None:
            return 0.0
        return float(np.clip(
            min(row.detection_rate, row.count_stability), 0.0, 1.0
        ))

    def presence_complete(self, entity_type: str) -> bool:
        row = self._entity(entity_type)
        if row is None:
            return False
        return bool(
            row.detection_rate >= self.minimum_detection_rate
            and row.mean_top_score >= self.minimum_presence_score
        )

    def presence_confidence(self, entity_type: str) -> float:
        """Presence completeness = fraction of normal images with a detection."""
        row = self._entity(entity_type)
        if row is None:
            return 0.0
        return float(np.clip(row.detection_rate, 0.0, 1.0))

    def metadata(self) -> Dict[str, Dict[str, Any]]:
        """Executor-facing maps; only calibrated entities appear as complete."""
        names = sorted(self.entities)
        return {
            "count_observation_complete": {
                name: self.count_complete(name) for name in names
            },
            "count_observation_confidence": {
                name: self.count_confidence(name) for name in names
            },
            "presence_observation_complete": {
                name: self.presence_complete(name) for name in names
            },
            "presence_observation_confidence": {
                name: self.presence_confidence(name) for name in names
            },
            "calibration_split": self.split,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "source_image_count": len(self.source_images),
            "minimum_detection_rate": self.minimum_detection_rate,
            "minimum_count_stability": self.minimum_count_stability,
            "minimum_presence_score": self.minimum_presence_score,
            "entities": {
                name: self.entities[name].as_dict() for name in sorted(self.entities)
            },
            "count_complete": {
                name: self.count_complete(name) for name in sorted(self.entities)
            },
            "presence_complete": {
                name: self.presence_complete(name) for name in sorted(self.entities)
            },
        }


def calibrate_from_counts(
    counts_per_image: Sequence[Mapping[str, Sequence[float]]],
    *,
    split: str,
    source_images: Sequence[str] = (),
    minimum_detection_rate: float = 0.9,
    minimum_count_stability: float = 0.9,
    minimum_presence_score: float = 0.3,
    explicit_entities: Iterable[str] = (),
) -> CountCalibration:
    """Estimate calibration from ``[{entity: (count, top_score)}, ...]``.

    ``explicit_entities`` must list super-types (e.g. ``fruit``) that were
    resolved through ``SceneGraph.type_hierarchy`` and therefore already have
    counts; without this the calibration silently omits them.
    """
    if not counts_per_image:
        raise ValueError("calibration needs at least one normal image")

    names = set(explicit_entities)
    for row in counts_per_image:
        names.update(row)
    entities: Dict[str, EntityCalibration] = {}
    for name in sorted(names):
        counts = []
        scores = []
        for row in counts_per_image:
            count, score = row.get(name, (0, 0.0))
            counts.append(int(count))
            scores.append(float(score))
        count_array = np.asarray(counts, dtype=int)
        score_array = np.asarray(scores, dtype=float)
        detection_rate = float(np.mean(count_array >= 1))
        modal_count = int(Counter(counts).most_common(1)[0][0])
        count_stability = float(np.mean(count_array == modal_count))
        present_scores = score_array[count_array >= 1]
        mean_top_score = (
            float(np.mean(present_scores)) if present_scores.size else 0.0
        )
        entities[name] = EntityCalibration(
            entity_type=name,
            sample_count=len(counts),
            detection_rate=detection_rate,
            count_stability=count_stability,
            modal_count=modal_count,
            mean_top_score=mean_top_score,
            count_min=int(count_array.min()),
            count_max=int(count_array.max()),
        )
    return CountCalibration(
        entities=entities,
        split=split,
        source_images=tuple(str(Path(p).resolve()) for p in source_images),
        minimum_detection_rate=minimum_detection_rate,
        minimum_count_stability=minimum_count_stability,
        minimum_presence_score=minimum_presence_score,
    )


__all__ = ["EntityCalibration", "CountCalibration", "calibrate_from_counts"]
