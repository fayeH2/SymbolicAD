"""Component segmentation using SAM (Segment Anything Model)."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from PIL import Image

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


@dataclass
class ComponentInstance:
    """Represents a detected component/part instance."""
    mask: np.ndarray  # Binary mask [H, W]
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    centroid: Tuple[float, float]  # normalized [0, 1]
    area: float  # normalized [0, 1]
    features: Dict[str, Any] = field(default_factory=dict)
    part_type: Optional[str] = None
    confidence: float = 1.0
    dino_features: Optional[np.ndarray] = None  # [C] or [H, W, C]


class ComponentSegmenter:
    """Segment components using SAM with optional DINOv2 feature extraction."""

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cuda:0",
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.5,
        stability_score_thresh: float = 0.92,
        box_nms_thresh: float = 0.7,
    ):
        self.device = device
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.mask_generator = None
        self.dino_model = None
        self.dino_transform = None
        self.stability_score_thresh = stability_score_thresh
        self.box_nms_thresh = box_nms_thresh

    def _load_sam(self):
        """Lazy load SAM model."""
        if self.mask_generator is not None:
            return
        if not SAM_AVAILABLE:
            raise RuntimeError("segment-anything not installed. Install with: pip install segment-anything")
        if self.sam_checkpoint is None or not Path(self.sam_checkpoint).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")
        
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            box_nms_thresh=self.box_nms_thresh,
        )

    def _load_dino(self):
        """Lazy load DINOv2 for feature extraction."""
        if self.dino_model is not None:
            return
        import torch
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.dino_model.to(self.device).eval()
        from torchvision import transforms
        self.dino_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.CenterCrop(392),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_features: bool = False,
    ) -> List[ComponentInstance]:
        """Segment image into component instances."""
        # Load image
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                rgb = np.array(img.convert("RGB"))
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = np.array(image)
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            elif rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

        h, w = rgb.shape[:2]
        self._load_sam()
        masks = self.mask_generator.generate(rgb)

        # Filter masks by area
        image_area = h * w
        filtered = []
        for m in masks:
            area_ratio = m["area"] / image_area
            if self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                filtered.append(m)

        # Extract component instances
        components = []
        for m in filtered:
            mask = m["segmentation"].astype(bool)
            x0, y0, bw, bh = m["bbox"]
            x1, y1 = x0 + bw, y0 + bh
            cx, cy = m["point_coords"][0] if m["point_coords"] else (x0 + bw/2, y0 + bh/2)
            cx, cy = cx / w, cy / h

            instance = ComponentInstance(
                mask=mask,
                bbox=(x0, y0, x1, y1),
                centroid=(cx, cy),
                area=area_ratio,
            )
            components.append(instance)

        # Sort by area descending
        components.sort(key=lambda c: c.area, reverse=True)

        # Extract attributes
        for comp in components:
            comp.features = extract_attributes(rgb, comp)

        # Extract DINOv2 features if requested
        if return_features:
            self._load_dino()
            import torch
            with torch.no_grad():
                for comp in components:
                    # Crop and extract features
                    crop = rgb[comp.bbox[1]:comp.bbox[3], comp.bbox[0]:comp.bbox[2]]
                    if crop.size > 0:
                        pil_crop = Image.fromarray(crop).resize((392, 392))
                        tensor = self.dino_transform(pil_crop).unsqueeze(0).to(self.device)
                        features = self.dino_model.forward_features(tensor)
                        # Use CLS token or patch features
                        comp.dino_features = features["x_norm_clstoken"].cpu().numpy().squeeze()

        return components


def extract_attributes(image: np.ndarray, instance: ComponentInstance) -> Dict[str, Any]:
    """Extract visual attributes for a component instance."""
    mask = instance.mask
    if not np.any(mask):
        return {}

    # Get masked region
    x0, y0, x1, y1 = instance.bbox
    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    if crop.size == 0:
        return {}

    # Color attributes (HSV)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    masked_hsv = hsv[crop_mask]
    if len(masked_hsv) > 0:
        mean_h = float(np.mean(masked_hsv[:, 0]))
        mean_s = float(np.mean(masked_hsv[:, 1]))
        mean_v = float(np.mean(masked_hsv[:, 2]))
        std_h = float(np.std(masked_hsv[:, 0]))
        std_s = float(np.std(masked_hsv[:, 1]))
    else:
        mean_h = mean_s = mean_v = std_h = std_s = 0.0

    # Shape attributes
    contours, _ = cv2.findContours(
        mask[y0:y1, x0:x1].astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / max(h, 1)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        solidity = area / hull_area if hull_area > 0 else 0
    else:
        area = perimeter = aspect_ratio = circularity = solidity = 0.0

    return {
        "color_h": mean_h, "color_s": mean_s, "color_v": mean_v,
        "color_h_std": std_h, "color_s_std": std_s,
        "shape_area": area, "shape_perimeter": perimeter,
        "shape_aspect": aspect_ratio, "shape_circularity": circularity,
        "shape_solidity": solidity,
        "bbox_width": x1 - x0, "bbox_height": y1 - y0,
    }


class FallbackSegmenter:
    """Fallback segmenter using classical CV when SAM is unavailable."""

    def __init__(self, min_area_ratio: float = 0.001, max_area_ratio: float = 0.5):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def segment(self, image: Union[str, Path, Image.Image, np.ndarray], return_features: bool = False) -> List[ComponentInstance]:
        """Segment using thresholding and connected components."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                rgb = np.array(img.convert("RGB"))
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = np.array(image)

        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        
        components = []
        image_area = h * w
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            area_ratio = area / image_area
            if self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                cx, cy = cx / w, cy / h
                mask = (labels == i)
                
                instance = ComponentInstance(
                    mask=mask,
                    bbox=(x, y, x + bw, y + bh),
                    centroid=(cx, cy),
                    area=area_ratio,
                )
                instance.features = extract_attributes(rgb, instance)
                components.append(instance)
        
        components.sort(key=lambda c: c.area, reverse=True)
        return components