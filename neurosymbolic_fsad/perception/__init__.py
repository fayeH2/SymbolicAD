"""Perception package for component detection and scene understanding."""

from neurosymbolic_fsad.perception.segmentation import ComponentSegmenter, ComponentInstance, FallbackSegmenter, extract_attributes
from neurosymbolic_fsad.perception.scene_graph import (
    SceneGraph, SpatialRelation, TopologicalRelation,
    build_scene_graph, align_scene_graphs, compute_normal_slots
)
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank, PartPrototype, match_components_to_prototypes
from neurosymbolic_fsad.perception.attributes import ComponentAttributes, compute_attribute_statistics

__all__ = [
    "ComponentSegmenter", "ComponentInstance", "FallbackSegmenter", "extract_attributes",
    "SceneGraph", "SpatialRelation", "TopologicalRelation",
    "build_scene_graph", "align_scene_graphs", "compute_normal_slots",
    "SemanticPrototypeBank", "PartPrototype", "match_components_to_prototypes",
    "ComponentAttributes", "compute_attribute_statistics",
]