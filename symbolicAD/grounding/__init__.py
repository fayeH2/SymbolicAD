"""正式空间归因接口；GT mask 只允许在 evaluation 包中出现。"""

from .counterfactual_repair import RepairMap, length_endpoint_repair
from .instance_grounder import InstanceGrounder, InstanceGrounding
from .relation_grounder import relation_witness_map
from .shape_grounder import length_violation_map, resized_instance_mask, union_instance_masks
from .slot_grounder import SlotGrounder, SlotPrototype

__all__ = [
    "RepairMap", "length_endpoint_repair", "InstanceGrounder",
    "InstanceGrounding", "relation_witness_map", "length_violation_map",
    "resized_instance_mask", "union_instance_masks", "SlotGrounder",
    "SlotPrototype",
]
