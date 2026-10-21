"""Canonical perception interfaces for component and scene understanding."""

from symbolicAD.perception.segmentation import ComponentSegmenter, FallbackSegmenter
from symbolicAD.perception.sam_dino_segmenter import SamDinoSegmenter, SamDinoSegmenterWrapper, create_sam_dino_segmenter, SAMDinoConfig, DinoV2FeatureExtractor
from symbolicAD.perception.component_instances import (
    ComponentInstance, instances_from_binary_mask, instances_from_label_mask,
)
from symbolicAD.perception.part_attributes import part_attributes, measure_mask_shape
from symbolicAD.perception.scene_graph import (
    SceneGraph, SpatialRelation, TopologicalRelation, RelationEdge,
    build_scene_graph, align_scene_graphs, compute_normal_slots
)
from symbolicAD.perception.prototypes import SemanticPrototypeBank, PartPrototype, match_components_to_prototypes
from symbolicAD.perception.attributes import ComponentAttributes, compute_attribute_statistics
from symbolicAD.perception.spatial_alignment import (
    AlignmentMatch, AlignmentResult, align_instances,
)

__all__ = [
    "ComponentSegmenter", "ComponentInstance", "FallbackSegmenter",
    "SamDinoSegmenter", "SamDinoSegmenterWrapper", "create_sam_dino_segmenter", "SAMDinoConfig", "DinoV2FeatureExtractor",
    "instances_from_binary_mask", "instances_from_label_mask",
    "part_attributes", "measure_mask_shape",
    "SceneGraph", "SpatialRelation", "TopologicalRelation", "RelationEdge",
    "build_scene_graph", "align_scene_graphs", "compute_normal_slots",
    "SemanticPrototypeBank", "PartPrototype", "match_components_to_prototypes",
    "ComponentAttributes", "compute_attribute_statistics",
    "AlignmentMatch", "AlignmentResult", "align_instances",
]
