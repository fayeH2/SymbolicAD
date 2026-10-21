"""MVTec LOCO dataset support for logical anomaly detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np

from symbolicAD.data import collect_test_samples, normal_train_images, select_inpformer_support


LOCO_CLASSES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

LOCO_LOGICAL_DEFECTS = {
    "breakfast_box": ["logical_anomalies"],
    "juice_bottle": ["logical_anomalies"],
    "pushpins": ["logical_anomalies"],
    "screw_bag": ["logical_anomalies"],
    "splicing_connectors": ["logical_anomalies"],
}

LOCO_STRUCTURAL_DEFECTS = {
    "breakfast_box": ["structural_anomalies"],
    "juice_bottle": ["structural_anomalies"],
    "pushpins": ["structural_anomalies"],
    "screw_bag": ["structural_anomalies"],
    "splicing_connectors": ["structural_anomalies"],
}


def get_loco_data_root() -> Path:
    """Get MVTec LOCO dataset root."""
    return Path("/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO")


def select_loco_support(
    data_root: Path,
    class_name: str,
    shot: int,
    seed: int = 668,
) -> List[Path]:
    """Select K-shot normal supports for MVTec LOCO.
    
    Uses only train/good images (excludes _val images from validation split).
    """
    class_root = data_root / class_name
    candidates = [
        p for p in normal_train_images(class_root)
        if not p.stem.endswith("_val")
    ]
    
    if shot < 1 or shot > len(candidates):
        raise ValueError(
            f"shot={shot}, but {len(candidates)} train-only normal images "
            f"exist for MVTec LOCO class {class_name}"
        )
    
    # Nested prefix sampling for consistent K=1/2/4/8
    import hashlib
    import random
    
    token = f"inp-support-v2:mvtec_loco:{class_name}:{seed}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    ordered = list(candidates)
    random.Random(derived_seed).shuffle(ordered)
    return ordered[:shot]


def collect_loco_test_samples(
    class_root: Path,
    test_subdir: str = "test",
    defect_type: str = "all",  # "logical", "structural", "all"
) -> List[Dict[str, Any]]:
    """Collect test samples from MVTec LOCO with defect type filtering."""
    samples = collect_test_samples(class_root, test_subdir)
    
    if defect_type == "logical":
        # Filter to only logical anomalies
        filtered = []
        for s in samples:
            defect = s["defect_type"]
            if "logical" in defect.lower():
                filtered.append(s)
        return filtered
    elif defect_type == "structural":
        filtered = []
        for s in samples:
            defect = s["defect_type"]
            if "structural" in defect.lower() or "good" in defect.lower():
                filtered.append(s)
        return filtered
    
    return samples


def get_loco_defect_types(class_name: str) -> Dict[str, List[str]]:
    """Get logical and structural defect types for a LOCO class."""
    class_root = get_loco_data_root() / class_name
    test_root = class_root / "test"
    
    logical = []
    structural = []
    
    if test_root.exists():
        for defect_dir in test_root.iterdir():
            if defect_dir.is_dir():
                name = defect_dir.name.lower()
                if "logical" in name:
                    logical.append(defect_dir.name)
                elif "structural" in name:
                    structural.append(defect_dir.name)
                elif name in ["good", "ok"]:
                    structural.append(defect_dir.name)
    
    return {"logical": logical, "structural": structural}


class LOCOEvaluator:
    """Evaluator for MVTec LOCO with separate logical/structural metrics."""
    
    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or get_loco_data_root()
    
    def evaluate(
        self,
        class_name: str,
        predictions: Dict[str, np.ndarray],  # path -> anomaly_map
        image_scores: Dict[str, float],       # path -> image_score
        defect_type: str = "all",
    ) -> Dict[str, Any]:
        """Evaluate predictions on LOCO dataset."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        class_root = self.data_root / class_name
        samples = collect_loco_test_samples(class_root, defect_type=defect_type)
        
        image_labels = []
        image_preds = []
        pixel_labels = []
        pixel_preds = []
        
        defect_types = []
        
        for sample in samples:
            path_str = str(sample["path"])
            label = sample["label"]
            defect = sample["defect_type"]
            
            if path_str not in predictions:
                continue
            
            anomaly_map = predictions[path_str]
            score = image_scores.get(path_str, float(np.max(anomaly_map)))
            
            # Get ground truth mask
            if label == 0:
                gt_mask = np.zeros_like(anomaly_map, dtype=bool)
            else:
                # Find mask file
                mask_path = self._find_mask(class_root, sample)
                if mask_path:
                    gt_mask = self._load_mask(mask_path, anomaly_map.shape)
                else:
                    gt_mask = np.zeros_like(anomaly_map, dtype=bool)
            
            image_labels.append(label)
            image_preds.append(score)
            pixel_labels.append(gt_mask.flatten())
            pixel_preds.append(anomaly_map.flatten())
            defect_types.append(defect)
        
        if not image_labels:
            return {"error": "No matching predictions found"}
        
        image_labels = np.array(image_labels)
        image_preds = np.array(image_preds)
        pixel_labels = np.concatenate(pixel_labels)
        pixel_preds = np.concatenate(pixel_preds)
        defect_types = np.array(defect_types)
        
        # Overall metrics
        metrics = {
            "image_auroc": float(roc_auc_score(image_labels, image_preds)),
            "image_ap": float(average_precision_score(image_labels, image_preds)),
            "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_preds)),
            "pixel_ap": float(average_precision_score(pixel_labels, pixel_preds)),
        }
        
        # Per-defect-type metrics
        for defect in np.unique(defect_types):
            if defect.lower() in ["good", "ok"]:
                continue
            mask = defect_types == defect
            if np.sum(mask) > 1:
                sel_labels = image_labels[mask]
                sel_preds = image_preds[mask]
                if len(np.unique(sel_labels)) > 1:
                    metrics[f"{defect}_image_auroc"] = float(
                        roc_auc_score(sel_labels, sel_preds)
                    )
        
        # Logical vs Structural breakdown
        logical_mask = np.array(["logical" in d.lower() for d in defect_types])
        structural_mask = np.array(["structural" in d.lower() for d in defect_types])
        
        if np.any(logical_mask):
            metrics["logical_image_auroc"] = float(
                roc_auc_score(image_labels[logical_mask], image_preds[logical_mask])
            )
        if np.any(structural_mask):
            metrics["structural_image_auroc"] = float(
                roc_auc_score(image_labels[structural_mask], image_preds[structural_mask])
            )
        
        return metrics
    
    def _find_mask(self, class_root: Path, sample: Dict[str, Any]) -> Optional[Path]:
        """Find ground truth mask for a sample."""
        image_path = Path(sample["path"])
        defect = image_path.parent.name
        
        candidates = [
            class_root / "ground_truth" / defect / f"{image_path.stem}_mask.png",
            class_root / "ground_truth" / defect / f"{image_path.stem}.png",
        ]
        
        for c in candidates:
            if c.exists():
                return c
        return None
    
    def _load_mask(self, mask_path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
        """Load and resize mask."""
        import cv2
        from PIL import Image
        
        with Image.open(mask_path) as img:
            mask = np.array(img.convert("L"))
        
        mask = cv2.resize(
            mask, (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        return mask > 128