"""Canonical perception interfaces for component and scene understanding."""

from neurosymbolic_fsad.perception.segmentation import ComponentSegmenter, FallbackSegmenter
from neurosymbolic_fsad.perception.component_instances import (
    ComponentInstance, instances_from_binary_mask, instances_from_label_mask,
)
from neurosymbolic_fsad.perception.part_attributes import part_attributes, measure_mask_shape
from neurosymbolic_fsad.perception.scene_graph import (
    SceneGraph, SpatialRelation, TopologicalRelation, RelationEdge,
    build_scene_graph, align_scene_graphs, compute_normal_slots
)
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank, PartPrototype, match_components_to_prototypes
from neurosymbolic_fsad.perception.attributes import ComponentAttributes, compute_attribute_statistics
from neurosymbolic_fsad.perception.spatial_alignment import (
    AlignmentMatch, AlignmentResult, align_instances,
)

__all__ = [
    "ComponentSegmenter", "ComponentInstance", "FallbackSegmenter",
    "instances_from_binary_mask", "instances_from_label_mask",
    "part_attributes", "measure_mask_shape",
    "SceneGraph", "SpatialRelation", "TopologicalRelation", "RelationEdge",
    "build_scene_graph", "align_scene_graphs", "compute_normal_slots",
    "SemanticPrototypeBank", "PartPrototype", "match_components_to_prototypes",
    "ComponentAttributes", "compute_attribute_statistics",
    "AlignmentMatch", "AlignmentResult", "align_instances",
]
