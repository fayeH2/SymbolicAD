"""SAM + DINOv2 Semantic Segmenter for SymbolicAD.

Replaces FallbackSegmenter with:
1. SAM (Segment Anything Model) for instance segmentation
2. DINOv2 for feature extraction and semantic classification
3. Prototype matching for semantic part type assignment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    import torch.hub
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False

from symbolicAD.perception.segmentation import ComponentInstance


@dataclass
class SAMDinoConfig:
    """Configuration for SAM + DINOv2 segmenter."""
    sam_checkpoint: str = "/home/hjh/AgentModels/sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    dinov2_model: str = "dinov2_vitb14_reg"
    device: str = "cuda:0"
    
    # SAM parameters
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 100
    
    # Filtering
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.5
    min_component_area: int = 50
    
    # DINOv2 feature extraction
    feature_layer: int = -1  # Last layer
    use_cls_token: bool = True
    
    # Prototype matching
    prototype_match_threshold: float = 0.7
    use_dino_features_for_matching: bool = True


class DinoV2FeatureExtractor:
    """DINOv2 feature extractor for component classification."""
    
    def __init__(self, model_name: str = "dinov2_vitb14_reg", device: str = "cuda:0"):
        self.device = device
        self.model_name = model_name
        
        # Load DINOv2
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(device).eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.CenterCrop(392),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Also prepare a smaller transform for component crops
        self.crop_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def extract_global_features(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extract global image features (CLS token)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            # Get patch tokens and cls token
            features = self.model.forward_features(tensor)
            cls_token = features['x_norm_clstoken']  # [1, D]
            
        return cls_token.cpu().numpy().squeeze()  # [D]
    
    @torch.no_grad()
    def extract_patch_features(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extract patch-level features for dense correspondence."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            features = self.model.forward_features(tensor)
            # x_norm_patchtokens: [1, N_patches, D]
            patch_tokens = features['x_norm_patchtokens']
            
        return patch_tokens.cpu().numpy().squeeze(0)  # [N_patches, D]
    
    @torch.no_grad()
    def extract_component_features(self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract DINOv2 features for a specific component."""
        x0, y0, x1, y1 = bbox
        component_crop = image[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]
        
        if component_crop.size == 0:
            return np.zeros(768)  # DINOv2-B hidden dim
        
        # Apply mask to crop
        masked_crop = component_crop.copy()
        masked_crop[mask_crop == 0] = [128, 128, 128]  # Gray background
        
        pil_crop = Image.fromarray(masked_crop.astype(np.uint8))
        tensor = self.crop_transform(pil_crop).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            features = self.model.forward_features(tensor)
            cls_token = features['x_norm_clstoken']
            
        return cls_token.cpu().numpy().squeeze()  # [D]


class SamDinoSegmenter:
    """SAM + DINOv2 semantic segmenter for industrial anomaly detection."""
    
    def __init__(self, config: Optional[SAMDinoConfig] = None):
        self.config = config or SAMDinoConfig()
        
        if not SAM_AVAILABLE:
            raise RuntimeError("segment-anything not installed. Run: pip install segment-anything")
        if not DINOV2_AVAILABLE:
            raise RuntimeError("torch.hub not available for DINOv2")
        
        self.device = self.config.device
        self._load_sam()
        self._load_dino()
        self.prototype_bank: Dict[str, List[np.ndarray]] = {}
    
    def _load_sam(self):
        """Load SAM model."""
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_checkpoint)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            crop_n_layers=self.config.crop_n_layers,
            crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
            min_mask_region_area=self.config.min_mask_region_area,
        )
    
    def _load_dino(self):
        """Load DINOv2 model."""
        self.dino_extractor = DinoV2FeatureExtractor(
            model_name=self.config.dinov2_model,
            device=self.device
        )
    
    def segment(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_features: bool = True,
    ) -> List[ComponentInstance]:
        """Segment image into component instances with SAM + DINOv2 features."""
        # Load image
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                rgb = np.array(img.convert("RGB"))
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = image.copy()
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            elif rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        
        h, w = rgb.shape[:2]
        
        # Generate SAM masks
        masks_data = self.mask_generator.generate(rgb)
        
        # Filter masks by area
        image_area = h * w
        filtered_masks = []
        for m in masks_data:
            area_ratio = m["area"] / image_area
            if self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio:
                if m["area"] >= self.config.min_component_area:
                    filtered_masks.append(m)
        
        # Sort by area descending
        filtered_masks.sort(key=lambda x: x["area"], reverse=True)
        
        # Extract DINOv2 features for each mask
        components = []
        for i, m in enumerate(filtered_masks):
            mask = m["segmentation"].astype(bool)
            x0, y0, bw, bh = m["bbox"]
            x1, y1 = x0 + bw, y0 + bh
            cx, cy = m["point_coords"][0] if m["point_coords"] else (x0 + bw/2, y0 + bh/2)
            cx, cy = cx / w, cy / h
            area_ratio = m["area"] / image_area
            
            instance = ComponentInstance(
                mask=mask,
                bbox=(x0, y0, x1, y1),
                centroid=(cx, cy),
                area=area_ratio,
            )
            
            # Extract DINOv2 features
            if return_features:
                dino_feat = self.dino_extractor.extract_component_features(rgb, mask, instance.bbox)
                instance.dino_features = dino_feat
            
            # Extract visual attributes
            instance.features = self._extract_attributes(rgb, instance)
            components.append(instance)
        
        # Sort by area descending
        components.sort(key=lambda c: c.area, reverse=True)
        return components
    
    def _extract_attributes(self, image: np.ndarray, instance: ComponentInstance) -> Dict[str, Any]:
        """Extract visual attributes for a component."""
        mask = instance.mask
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
        
        # Length feature (approximate from major axis)
        length = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                length = max(ellipse[1]) / max(h, w)  # Normalized major axis
        
        return {
            "color_h": mean_h, "color_s": mean_s, "color_v": mean_v,
            "color_h_std": std_h, "color_s_std": std_s,
            "shape_area": float(area), "shape_perimeter": float(perimeter),
            "shape_aspect": aspect_ratio, "shape_circularity": circularity,
            "shape_solidity": solidity, "shape_length": length,
            "bbox_width": x1 - x0, "bbox_height": y1 - y0,
        }
    
    def build_prototype_bank(self, support_images: List[Union[str, Path, Image.Image, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """Build prototype bank from normal support images."""
        prototype_bank: Dict[str, List[np.ndarray]] = {}
        
        for img_path in support_images:
            components = self.segment(img_path, return_features=True)
            
            # Match components to existing prototypes or create new
            for comp in components:
                if comp.dino_features is None:
                    continue
                
                feat = comp.dino_features / (np.linalg.norm(comp.dino_features) + 1e-8)
                
                # Find best matching prototype
                best_type = None
                best_sim = -1
                
                for p_type, protos in self.prototype_bank.items():
                    for proto in protos:
                        proto_norm = proto / (np.linalg.norm(proto) + 1e-8)
                        sim = float(np.dot(feat, proto_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_type = p_type
                
                if best_sim > self.config.prototype_match_threshold:
                    self.prototype_bank[best_type].append(feat)
                else:
                    # Create new prototype type
                    new_type = f"part_{len(self.prototype_bank)}"
                    self.prototype_bank[new_type] = [feat]
        
        return self.prototype_bank
    
    def match_components_to_prototypes(
        self,
        components: List[ComponentInstance],
        threshold: float = 0.7
    ) -> Dict[int, str]:
        """Match components to prototype bank."""
        matches = {}
        
        for i, comp in enumerate(components):
            if comp.dino_features is None:
                matches[i] = "unknown"
                continue
            
            feat = comp.dino_features / (np.linalg.norm(comp.dino_features) + 1e-8)
            best_type = "unknown"
            best_sim = 0.0
            
            for p_type, protos in self.prototype_bank.items():
                for proto in protos:
                    proto_norm = proto / (np.linalg.norm(proto) + 1e-8)
                    sim = float(np.dot(feat, proto_norm))
                    if sim > best_sim:
                        best_sim = sim
                        best_type = p_type
            
            if best_sim >= threshold:
                matches[i] = best_type
            else:
                matches[i] = "unknown"
        
        return matches


def create_sam_dino_segmenter(
    sam_checkpoint: str = "/home/hjh/AgentModels/sam_vit_h_4b8939.pth",
    device: str = "cuda:0"
) -> SamDinoSegmenter:
    """Factory function to create SAM+DINOv2 segmenter."""
    config = SAMDinoConfig(
        sam_checkpoint=sam_checkpoint,
        device=device,
    )
    return SamDinoSegmenter(config)


# Compatibility wrapper to replace FallbackSegmenter
class SamDinoSegmenterWrapper:
    """Wrapper to make SamDinoSegmenter compatible with FallbackSegmenter interface."""
    
    def __init__(self, sam_checkpoint: str = "/home/hjh/AgentModels/sam_vit_h_4b8939.pth", device: str = "cuda:0"):
        self.segmenter = create_sam_dino_segmenter(sam_checkpoint, device)
    
    def segment(self, image: Union[str, Path, Image.Image, np.ndarray], return_features: bool = True) -> List[ComponentInstance]:
        return self.segmenter.segment(image, return_features=return_features)


# Export
__all__ = [
    "SAMDinoConfig", "SamDinoSegmenter", "DinoV2FeatureExtractor",
    "create_sam_dino_segmenter", "SamDinoSegmenterWrapper",
]