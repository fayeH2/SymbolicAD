"""规则事实层：开放词表检测 -> 观测校准 -> SceneGraph -> ExecutionContext。"""

from .calibration import (
    CountCalibration, EntityCalibration, calibrate_from_counts,
)
from .open_vocab import (
    Detection, DetectionBatch, GroundingDinoDetector, Vocabulary,
)
from .scene_builder import (
    DEFAULT_BREAKFAST_VOCABULARY, DEFAULT_SUPERTYPES, SceneFactBuilder,
    SceneFactConfig, batch_from_record, detection_record,
    estimate_vertical_divider,
)

__all__ = [
    "Detection", "DetectionBatch", "GroundingDinoDetector", "Vocabulary",
    "CountCalibration", "EntityCalibration", "calibrate_from_counts",
    "DEFAULT_BREAKFAST_VOCABULARY", "DEFAULT_SUPERTYPES", "SceneFactBuilder",
    "SceneFactConfig", "batch_from_record", "detection_record",
    "estimate_vertical_divider",
]

from .interventions import (
    EditedImage, PixelEdit, confirm_intervention, empty_region, erase_entity,
    flip_region_vertical, relocate_entity_to_region,
    separate_entity_horizontally,
)

__all__ += [
    "EditedImage", "PixelEdit", "confirm_intervention", "empty_region", "erase_entity",
    "flip_region_vertical", "relocate_entity_to_region",
    "separate_entity_horizontally",
]
