"""
Structured Perception V1 - 视觉特征结构化提取器

从 INP-Former 的 DINOv2 特征中提取结构化视觉信息：
- 区域发现 (patch聚类 + 连通域)
- 结构化属性: cluster_id, bbox, center, area, mean_color
- 空间关系: LeftOf, RightOf, Above, Below, Near, Far, Symmetric, Aligned
- 规则执行器: 支持 Exists, Count, LeftOf, RightOf, Near, Symmetric, Aligned

设计原则：
- 统一的视觉码本 (仅在正常 support 上拟合)
- 匿名聚类类别 (不预设语义标签)
- 统一的空间关系计算 (无需 CLIP/文本)
- 可序列化为 JSON 供 LLM 规则执行器使用
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class VisualRegion:
    """单个候选视觉区域的结构化属性"""
    region_id: str
    cluster_id: int
    area_patches: int
    area_ratio: float
    center_xy: Tuple[float, float]      # 归一化 [0,1] 中心坐标
    bbox_xyxy: Tuple[float, float, float, float]  # 归一化 [x1,y1,x2,y2]
    mean_color: Optional[Tuple[float, float, float]] = None  # RGB 平均颜色
    cluster_confidence: float = 1.0     # 该区域内 patch 的平均聚类置信度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "cluster_id": self.cluster_id,
            "area_patches": self.area_patches,
            "area_ratio": round(self.area_ratio, 4),
            "center_xy": [round(self.center_xy[0], 4), round(self.center_xy[1], 4)],
            "bbox_xyxy": [round(v, 4) for v in self.bbox_xyxy],
            "mean_color": [round(v, 4) for v in self.mean_color] if self.mean_color else None,
            "cluster_confidence": round(self.cluster_confidence, 4),
        }


@dataclass
class SpatialRelation:
    """两个区域之间的空间关系"""
    predicate: str              # LeftOf, RightOf, Above, Below, Near, Far, Symmetric, Aligned
    subject_id: str
    object_id: str
    value: float                # 连续值 [0,1] 或布尔 0/1
    threshold: float            # 判定阈值
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "holds": self.value >= self.threshold,
        }


@dataclass
class StructuredPrediction:
    """结构化感知的完整输出"""
    regions: List[VisualRegion]
    relations: List[SpatialRelation]
    candidate_region_counts: Dict[int, int]  # cluster_id -> count
    image_size: Tuple[int, int]  # (H, W) patch grid size
    
    def to_serializable(self) -> Dict[str, Any]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "relations": [r.to_dict() for r in self.relations],
            "candidate_region_counts": self.candidate_region_counts,
            "image_size": self.image_size,
            "num_regions": len(self.regions),
            "num_relations": len(self.relations),
        }


class VisualCodebook:
    """视觉码本：在正常 support 上拟合 K-means 聚类中心"""
    
    def __init__(
        self,
        n_clusters: int = 6,
        min_region_patches: int = 3,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.n_clusters = n_clusters
        self.min_region_patches = min_region_patches
        self.seed = seed
        self.device = device
        
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers: Optional[np.ndarray] = None  # [K, D] normalized
        self.fitted = False
    
    def fit(self, support_features: torch.Tensor) -> "VisualCodebook":
        """
        在正常 support 特征上拟合视觉码本
        
        Args:
            support_features: [K, C, H, W] 或 [N, C, H, W] - 正常 support 的编码器特征
        """
        # 展平为 [N_patches, C]
        if support_features.dim() == 4:
            B, C, H, W = support_features.shape
            features = support_features.permute(0, 2, 3, 1).reshape(-1, C)
            self.image_size = (H, W)
        else:
            features = support_features
            self.image_size = (28, 28)  # default
        
        # L2 归一化
        features_np = features.detach().cpu().numpy().astype(np.float32)
        features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
        
        # K-means 聚类
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10,
            max_iter=300,
        )
        self.kmeans.fit(features_norm)
        
        # 保存归一化的聚类中心
        self.cluster_centers = self.kmeans.cluster_centers_.astype(np.float32)
        self.cluster_centers = self.cluster_centers / (
            np.linalg.norm(self.cluster_centers, axis=1, keepdims=True) + 1e-8
        )
        
        # 关键：根据聚类中心在 support 上的空间分布对 cluster ID 进行稳定排序
        # 这样 cluster ID 就具有稳定的空间语义
        if hasattr(self, 'image_size') and self.image_size is not None:
            self._stabilize_cluster_ids(support_features)
        
        self.fitted = True
        return self

    def _stabilize_cluster_ids(self, support_features: torch.Tensor):
        """
        根据聚类在 support 上的空间分布稳定化 cluster ID
        
        策略：
        1. 计算每个聚类在 support 上的空间质心
        2. 根据空间位置排序：先按 y 坐标（从上到下），再按 x 坐标（从左到右）
        3. 重新分配 cluster ID
        """
        B, C, H, W = support_features.shape
        features = support_features.permute(0, 2, 3, 1).reshape(-1, C)
        features_norm = features.detach().cpu().numpy().astype(np.float32)
        features_norm = features_norm / (np.linalg.norm(features_norm, axis=1, keepdims=True) + 1e-8)
        
        # 使用当前的聚类中心预测 support 上的聚类分配
        similarities = features_norm @ self.cluster_centers.T  # [N, K]
        cluster_ids = similarities.argmax(axis=1)
        
        # 计算每个 cluster 在 support 上的空间质心
        cluster_spatial_info = []
        patches_per_image = H * W
        for k in range(self.n_clusters):
            mask = (cluster_ids == k)
            if not mask.any():
                # 空 cluster，给一个默认位置
                cluster_spatial_info.append((k, 1.0, 1.0, 0))
                continue
            
            # 计算该 cluster 所有 patch 的平均坐标
            patch_indices = np.where(mask)[0]
            # 处理 batch 维度：先确定属于哪张图，再计算图内坐标
            img_indices = patch_indices // (H * W)
            patch_in_img = patch_indices % (H * W)
            y_coords = patch_in_img // W
            x_coords = patch_in_img % W
            cy = y_coords.mean() / (H - 1)  # 归一化 [0,1]
            cx = x_coords.mean() / (W - 1)
            area = mask.sum()
            
            cluster_spatial_info.append((k, cy, cx, area))
        
        # 根据空间位置排序：先按 y (从上到下)，再按 x (从左到右)
        cluster_spatial_info.sort(key=lambda x: (x[1], x[2]))
        
        # 建立旧 ID -> 新 ID 的映射
        old_to_new = {old_id: new_id for new_id, (old_id, _, _, _) in enumerate(cluster_spatial_info)}
        
        # 重新排序 cluster_centers
        new_centers = np.zeros_like(self.cluster_centers)
        for old_id, new_id in old_to_new.items():
            new_centers[new_id] = self.cluster_centers[old_id]
        
        self.cluster_centers = new_centers
        self.old_to_new_id = old_to_new
        self.cluster_spatial_info = [(cy, cx, area) for _, cy, cx, area in cluster_spatial_info]
    
    def predict(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测特征的聚类分配和置信度
        
        Returns:
            cluster_ids: [N_patches] cluster index
            confidences: [N_patches] cosine similarity to assigned center
        """
        if not self.fitted:
            raise RuntimeError("Codebook not fitted. Call fit() first.")
        
        if features.dim() == 4:
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            features_flat = features
        
        features_np = features_flat.detach().cpu().numpy().astype(np.float32)
        features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
        
        # 余弦相似度分配
        similarities = features_norm @ self.cluster_centers.T  # [N, K]
        cluster_ids = similarities.argmax(axis=1)
        confidences = similarities.max(axis=1)
        
        return cluster_ids, confidences
    
    def predict_map(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """预测特征图的聚类图和置信度图"""
        if features.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got {features.shape}")
        
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        cluster_ids, confidences = self.predict(features_flat)
        
        cluster_map = cluster_ids.reshape(B, H, W)
        conf_map = confidences.reshape(B, H, W)
        
        return cluster_map, conf_map


class StructuredPerception:
    """结构化感知提取器 V1"""
    
    def __init__(
        self,
        n_clusters: int = 6,
        min_region_patches: int = 3,
        min_region_area_ratio: float = 0.001,
        spatial_relation_thresholds: Optional[Dict[str, float]] = None,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.n_clusters = n_clusters
        self.min_region_patches = min_region_patches
        self.min_region_area_ratio = min_region_area_ratio
        self.seed = seed
        self.device = device
        
        # 视觉码本
        self.codebook = VisualCodebook(
            n_clusters=n_clusters,
            min_region_patches=min_region_patches,
            seed=seed,
            device=device,
        )
        
        # 空间关系阈值
        self.thresholds = spatial_relation_thresholds or {
            "near": 0.15,      # 归一化距离阈值
            "far": 0.4,
            "aligned_tol": 0.05,  # 对齐容差
            "symmetric_tol": 0.08,  # 对称容差
        }
        
        self.fitted = False
        self.image_size: Optional[Tuple[int, int]] = None
        self._rgb_cache: Optional[np.ndarray] = None  # 用于 mean_color 计算
    
    def fit_support(self, support_features: torch.Tensor) -> "StructuredPerception":
        """在正常 support 上拟合视觉码本"""
        self.codebook.fit(support_features)
        self.fitted = True
        # 记录图像尺寸 (patch grid)
        if support_features.dim() == 4:
            _, _, H, W = support_features.shape
            self.image_size = (H, W)
        return self
    
    def set_rgb_cache(self, rgb_image: np.ndarray):
        """缓存原始 RGB 图像用于 mean_color 计算"""
        self._rgb_cache = rgb_image
    
    def extract_regions(
        self,
        query_features: torch.Tensor,
        rgb_image: Optional[np.ndarray] = None,
    ) -> List[VisualRegion]:
        """从查询特征中提取候选区域"""
        if not self.fitted:
            raise RuntimeError("Call fit_support() first.")
        
        if query_features.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got {query_features.shape}")
        
        B, C, H, W = query_features.shape
        self.image_size = (H, W)
        
        # 预测聚类图
        cluster_map, conf_map = self.codebook.predict_map(query_features)
        
        # 缓存 RGB
        if rgb_image is not None:
            self._rgb_cache = rgb_image
        
        regions = []
        
        for b in range(B):
            b_cluster = cluster_map[b]  # [H, W]
            b_conf = conf_map[b]        # [H, W]
            
            # 对每个聚类类别做连通域分析
            for cluster_id in range(self.n_clusters):
                mask = (b_cluster == cluster_id)
                if not mask.any():
                    continue
                
                # 连通域标记
                labeled, num_components = nd_label(mask)
                
                for comp_id in range(1, num_components + 1):
                    comp_mask = (labeled == comp_id)
                    area_patches = comp_mask.sum()
                    
                    # 过滤太小的区域
                    if area_patches < self.min_region_patches:
                        continue
                    
                    area_ratio = area_patches / (H * W)
                    if area_ratio < self.min_region_area_ratio:
                        continue
                    
                    # 计算边界框和中心
                    y_indices, x_indices = np.where(comp_mask)
                    y1, y2 = y_indices.min(), y_indices.max()
                    x1, x2 = x_indices.min(), x_indices.max()
                    
                    # 归一化坐标 [0,1]
                    cx = (x1 + x2) / 2 / (W - 1)
                    cy = (y1 + y2) / 2 / (H - 1)
                    nx1, nx2 = x1 / (W - 1), x2 / (W - 1)
                    ny1, ny2 = y1 / (H - 1), y2 / (H - 1)
                    
                    # 计算置信度
                    cluster_conf = b_conf[comp_mask].mean()
                    
                    # 计算平均颜色
                    mean_color = None
                    if self._rgb_cache is not None:
                        # 将 patch 坐标映射回原图
                        rgb_h, rgb_w = self._rgb_cache.shape[:2]
                        scale_h, scale_w = rgb_h / H, rgb_w / W
                        y1_rgb = int(y1 * scale_h)
                        y2_rgb = int((y2 + 1) * scale_h)
                        x1_rgb = int(x1 * scale_w)
                        x2_rgb = int((x2 + 1) * scale_w)
                        y1_rgb = max(0, min(y1_rgb, rgb_h - 1))
                        y2_rgb = max(0, min(y2_rgb, rgb_h))
                        x1_rgb = max(0, min(x1_rgb, rgb_w - 1))
                        x2_rgb = max(0, min(x2_rgb, rgb_w))
                        if y2_rgb > y1_rgb and x2_rgb > x1_rgb:
                            patch_rgb = self._rgb_cache[y1_rgb:y2_rgb, x1_rgb:x2_rgb]
                            mean_color = tuple(patch_rgb.mean(axis=(0, 1)).astype(float))
                    
                    region_id = f"cluster{cluster_id}_region{len(regions)}"
                    
                    regions.append(VisualRegion(
                        region_id=region_id,
                        cluster_id=cluster_id,
                        area_patches=int(area_patches),
                        area_ratio=float(area_ratio),
                        center_xy=(float(cx), float(cy)),
                        bbox_xyxy=(float(nx1), float(ny1), float(nx2), float(ny2)),
                        mean_color=mean_color,
                        cluster_confidence=float(cluster_conf),
                    ))
        
        return regions
    
    def compute_spatial_relations(self, regions: List[VisualRegion]) -> List[SpatialRelation]:
        """计算所有区域对之间的空间关系"""
        relations = []
        n = len(regions)
        
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = regions[i], regions[j]
                cx1, cy1 = r1.center_xy
                cx2, cy2 = r2.center_xy
                
                dx = cx2 - cx1
                dy = cy2 - cy1
                dist = math.hypot(dx, dy)
                
                # LeftOf / RightOf
                if dx > self.thresholds.get("near", 0.15):
                    relations.append(SpatialRelation(
                        predicate="LeftOf", subject_id=r1.region_id, object_id=r2.region_id,
                        value=1.0, threshold=0.5
                    ))
                    relations.append(SpatialRelation(
                        predicate="RightOf", subject_id=r2.region_id, object_id=r1.region_id,
                        value=1.0, threshold=0.5
                    ))
                elif dx < -self.thresholds.get("near", 0.15):
                    relations.append(SpatialRelation(
                        predicate="RightOf", subject_id=r1.region_id, object_id=r2.region_id,
                        value=1.0, threshold=0.5
                    ))
                    relations.append(SpatialRelation(
                        predicate="LeftOf", subject_id=r2.region_id, object_id=r1.region_id,
                        value=1.0, threshold=0.5
                    ))
                
                # Above / Below
                if dy > self.thresholds.get("near", 0.15):
                    relations.append(SpatialRelation(
                        predicate="Above", subject_id=r1.region_id, object_id=r2.region_id,
                        value=1.0, threshold=0.5
                    ))
                    relations.append(SpatialRelation(
                        predicate="Below", subject_id=r2.region_id, object_id=r1.region_id,
                        value=1.0, threshold=0.5
                    ))
                elif dy < -self.thresholds.get("near", 0.15):
                    relations.append(SpatialRelation(
                        predicate="Below", subject_id=r1.region_id, object_id=r2.region_id,
                        value=1.0, threshold=0.5
                    ))
                    relations.append(SpatialRelation(
                        predicate="Above", subject_id=r2.region_id, object_id=r1.region_id,
                        value=1.0, threshold=0.5
                    ))
                
                # Near / Far (连续值)
                near_val = max(0.0, 1.0 - dist / self.thresholds.get("near", 0.15))
                far_val = min(1.0, dist / self.thresholds.get("far", 0.4))
                
                relations.append(SpatialRelation(
                    predicate="Near", subject_id=r1.region_id, object_id=r2.region_id,
                    value=near_val, threshold=self.thresholds.get("near", 0.15)
                ))
                relations.append(SpatialRelation(
                    predicate="Far", subject_id=r1.region_id, object_id=r2.region_id,
                    value=far_val, threshold=self.thresholds.get("far", 0.4)
                ))
                
                # Aligned (水平/垂直对齐)
                aligned_tol = self.thresholds.get("aligned_tol", 0.05)
                if abs(cy1 - cy2) < aligned_tol:
                    relations.append(SpatialRelation(
                        predicate="AlignedH", subject_id=r1.region_id, object_id=r2.region_id,
                        value=1.0 - abs(cy1 - cy2) / aligned_tol, threshold=0.5
                    ))
                if abs(cx1 - cx2) < aligned_tol:
                    relations.append(SpatialRelation(
                        predicate="AlignedV", subject_id=r1.region_id, object_id=r2.region_id,
                        value=1.0 - abs(cx1 - cx2) / aligned_tol, threshold=0.5
                    ))
                
                # Symmetric (关于图像中心对称)
                sym_cx1, sym_cy1 = 1 - cx1, 1 - cy1
                sym_dist = math.hypot(sym_cx1 - cx2, sym_cy1 - cy2)
                sym_val = max(0.0, 1.0 - sym_dist / self.thresholds.get("symmetric_tol", 0.08))
                relations.append(SpatialRelation(
                    predicate="Symmetric", subject_id=r1.region_id, object_id=r2.region_id,
                    value=sym_val, threshold=self.thresholds.get("symmetric_tol", 0.08)
                ))
        
        return relations
    
    def count_regions_by_cluster(self, regions: List[VisualRegion]) -> Dict[int, int]:
        """统计每个聚类的区域数量"""
        counts = {}
        for r in regions:
            counts[r.cluster_id] = counts.get(r.cluster_id, 0) + 1
        return counts
    
    def predict(
        self,
        query_features: torch.Tensor,
        rgb_image: Optional[np.ndarray] = None,
    ) -> StructuredPrediction:
        """完整的结构化感知流程"""
        if not self.fitted:
            raise RuntimeError("Call fit_support() first.")
        
        # 1. 提取候选区域
        regions = self.extract_regions(query_features, rgb_image)
        
        # 2. 计算空间关系
        relations = self.compute_spatial_relations(regions)
        
        # 3. 统计区域计数
        counts = self.count_regions_by_cluster(regions)
        
        return StructuredPrediction(
            regions=regions,
            relations=relations,
            candidate_region_counts=counts,
            image_size=self.image_size or (0, 0),
        )
    
    def to_serializable(self, prediction: StructuredPrediction) -> Dict[str, Any]:
        return prediction.to_serializable()


# ===== 规则执行器 V1 =====

class RuleExecutor:
    """简单规则执行器：基于结构化感知输出执行逻辑规则"""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "near": 0.15,
            "far": 0.4,
            "aligned_tol": 0.05,
            "symmetric_tol": 0.08,
            "exists_min_area": 0.001,
        }
    
    def evaluate(
        self,
        prediction: StructuredPrediction,
        rule: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行单条规则
        
        Rule 格式示例：
        {
            "type": "Exists",
            "cluster_id": 2,
            "min_count": 1,
        }
        {
            "type": "LeftOf",
            "subject_cluster": 1,
            "object_cluster": 2,
        }
        {
            "type": "Symmetric",
            "cluster_id": 0,
            "tolerance": 0.08
        }
        """
        rule_type = rule.get("type", "").lower()
        
        # 构建区域索引
        regions_by_cluster: Dict[int, List[VisualRegion]] = {}
        for r in prediction.regions:
            regions_by_cluster.setdefault(r.cluster_id, []).append(r)
        
        region_map = {r.region_id: r for r in prediction.regions}
        
        if rule_type == "exists":
            cluster_id = rule.get("cluster_id")
            min_count = rule.get("min_count", 1)
            count = len(regions_by_cluster.get(cluster_id, []))
            satisfied = count >= min_count
            return {
                "satisfied": satisfied,
                "value": float(count >= min_count),
                "detail": f"Cluster {cluster_id}: {count} regions, need >= {min_count}",
            }
        
        elif rule_type == "count":
            cluster_id = rule.get("cluster_id")
            expected = rule.get("expected")
            op = rule.get("op", "==")
            count = len(regions_by_cluster.get(cluster_id, []))
            
            if op == "==":
                satisfied = count == expected
            elif op == "!=":
                satisfied = count != expected
            elif op == ">":
                satisfied = count > expected
            elif op == "<":
                satisfied = count < expected
            elif op == ">=":
                satisfied = count >= expected
            elif op == "<=":
                satisfied = count <= expected
            else:
                satisfied = False
            
            return {
                "satisfied": satisfied,
                "value": float(satisfied),
                "detail": f"Cluster {cluster_id}: {count} {op} {expected}",
            }
        
        elif rule_type in ("leftof", "rightof", "above", "below"):
            subj_cluster = rule.get("subject_cluster")
            obj_cluster = rule.get("object_cluster")
            
            subj_regions = regions_by_cluster.get(subj_cluster, [])
            obj_regions = regions_by_cluster.get(obj_cluster, [])
            
            if not subj_regions or not obj_regions:
                return {"satisfied": False, "value": 0.0, "detail": "Missing regions"}
            
            # 任意一对满足即视为满足
            satisfied = False
            best_val = 0.0
            
            for sr in subj_regions:
                for or_ in obj_regions:
                    cx1, cy1 = sr.center_xy
                    cx2, cy2 = or_.center_xy
                    dx = cx2 - cx1
                    dy = cy2 - cy1
                    
                    if rule_type == "leftof":
                        val = 1.0 if dx > 0.1 else max(0.0, dx / 0.15)
                        if dx > 0.1: satisfied = True
                    elif rule_type == "rightof":
                        val = 1.0 if dx < -0.1 else max(0.0, -dx / 0.15)
                        if dx < -0.1: satisfied = True
                    elif rule_type == "above":
                        val = 1.0 if dy < -0.1 else max(0.0, -dy / 0.15)
                        if dy < -0.1: satisfied = True
                    elif rule_type == "below":
                        val = 1.0 if dy > 0.1 else max(0.0, dy / 0.15)
                        if dy > 0.1: satisfied = True
                    
                    if val > best_val:
                        best_val = val
            
            return {
                "satisfied": satisfied,
                "value": best_val,
                "detail": f"{rule_type}: {subj_cluster} vs {obj_cluster} = {best_val:.2f}",
            }
        
        elif rule_type == "near":
            cluster1 = rule.get("cluster1")
            cluster2 = rule.get("cluster2")
            threshold = rule.get("threshold", 0.15)
            
            r1_list = regions_by_cluster.get(cluster1, [])
            r2_list = regions_by_cluster.get(cluster2, [])
            
            satisfied = False
            best_val = 0.0
            
            for r1 in r1_list:
                for r2 in r2_list:
                    dist = math.hypot(r1.center_xy[0] - r2.center_xy[0], 
                                     r1.center_xy[1] - r2.center_xy[1])
                    val = max(0.0, 1.0 - dist / 0.15)
                    if dist < threshold:
                        satisfied = True
                    if val > best_val:
                        best_val = val
            
            return {
                "satisfied": satisfied,
                "value": best_val,
                "detail": f"Near: {cluster1}-{cluster2} = {best_val:.2f}",
            }
        
        elif rule_type == "symmetric":
            cluster_id = rule.get("cluster_id")
            tolerance = rule.get("tolerance", 0.08)
            
            regions = regions_by_cluster.get(cluster_id, [])
            if len(regions) < 2:
                return {"satisfied": False, "value": 0.0, "detail": "Need at least 2 regions"}
            
            # 检查是否存在对称对
            satisfied = False
            best_val = 0.0
            
            for i, r1 in enumerate(regions):
                for r2 in regions[i+1:]:
                    cx1, cy1 = r1.center_xy
                    cx2, cy2 = r2.center_xy
                    sym_cx1, sym_cy1 = 1 - cx1, 1 - cy1
                    sym_dist = math.hypot(sym_cx1 - cx2, sym_cy1 - cy2)
                    val = max(0.0, 1.0 - sym_dist / 0.08)
                    if sym_dist < tolerance:
                        satisfied = True
                    if val > best_val:
                        best_val = val
            
            return {
                "satisfied": satisfied,
                "value": best_val,
                "detail": f"Symmetric: {cluster_id} = {best_val:.2f}",
            }
        
        elif rule_type == "aligned":
            axis = rule.get("axis", "horizontal")  # horizontal or vertical
            cluster_id = rule.get("cluster_id")
            tolerance = rule.get("tolerance", 0.05)
            
            regions = regions_by_cluster.get(cluster_id, [])
            if len(regions) < 2:
                return {"satisfied": False, "value": 0.0, "detail": "Need at least 2 regions"}
            
            if axis == "horizontal":
                # 水平对齐：y坐标接近
                ys = [r.center_xy[1] for r in regions]
                spread = max(ys) - min(ys)
                val = max(0.0, 1.0 - spread / 0.05)
                satisfied = spread < tolerance
            else:
                # 垂直对齐：x坐标接近
                xs = [r.center_xy[0] for r in regions]
                spread = max(xs) - min(xs)
                val = max(0.0, 1.0 - spread / 0.05)
                satisfied = spread < tolerance
            
            return {
                "satisfied": satisfied,
                "value": val,
                "detail": f"Aligned{axis}: spread={spread:.3f}",
            }
        
        return {
            "satisfied": False,
            "value": 0.0,
            "detail": f"Unknown rule type: {rule_type}",
        }

    # ===== 新增鲁棒规则类型 =====
    
    def _evaluate_cluster_profile(self, prediction: StructuredPrediction, rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估簇的整体概况：存在性、数量、面积、位置分布
        这是最通用的规则类型，替代简单的 Exists/Count
        """
        profile = rule.get("profile", {})  # 预期的簇概况
        tolerance = rule.get("tolerance", 0.5)  # 容差
        
        # 计算实际概况
        actual = {}
        for cluster_id, count in prediction.candidate_region_counts.items():
            regions = [r for r in prediction.regions if r.cluster_id == cluster_id]
            if not regions:
                continue
            total_area = sum(r.area_patches for r in regions)
            avg_cx = sum(r.center_xy[0] for r in regions) / len(regions)
            avg_cy = sum(r.center_xy[1] for r in regions) / len(regions)
            actual[cluster_id] = {
                "count": count,
                "total_area": total_area,
                "avg_area": total_area / count,
                "center_xy": (avg_cx, avg_cy),
            }
        
        # 计算与预期的偏差
        total_deviation = 0.0
        max_deviation = 0.0
        details = []
        
        for cluster_id, expected in profile.items():
            if cluster_id not in actual:
                total_deviation += 1.0
                max_deviation = max(max_deviation, 1.0)
                details.append(f"Cluster {cluster_id}: missing (expected {expected})")
                continue
            
            act = actual[cluster_id]
            dev = 0.0
            # 数量偏差：支持范围 (min, max) 或单值
            if "count" in expected:
                exp_count = expected["count"]
                if isinstance(exp_count, tuple) and len(exp_count) == 2:
                    # 范围检查
                    min_c, max_c = exp_count
                    if act["count"] < min_c or act["count"] > max_c:
                        cnt_dev = min(abs(act["count"] - min_c), abs(act["count"] - max_c)) / max(1, (min_c + max_c) / 2)
                    else:
                        cnt_dev = 0.0  # 在范围内，偏差为0
                else:
                    cnt_dev = abs(act["count"] - exp_count) / max(1, exp_count)
                dev = max(dev, cnt_dev)
            # 面积偏差
            if "area" in expected:
                exp_area = expected["area"]
                if isinstance(exp_area, tuple) and len(exp_area) == 2:
                    min_a, max_a = exp_area
                    if act["total_area"] < min_a or act["total_area"] > max_a:
                        area_dev = min(abs(act["total_area"] - min_a), abs(act["total_area"] - max_a)) / max(1, (min_a + max_a) / 2)
                    else:
                        area_dev = abs(act["total_area"] - exp_area) / max(1, exp_area)
                else:
                    area_dev = abs(act["total_area"] - exp_area) / max(1, exp_area)
                dev = max(dev, area_dev)
            # 位置偏差
            if "center_xy" in expected:
                exp_cx, exp_cy = expected["center_xy"]
                act_cx, act_cy = act["center_xy"]
                pos_dev = math.hypot(act_cx - exp_cx, act_cy - exp_cy)
                dev = max(dev, pos_dev)
            
            total_deviation += dev
            max_deviation = max(max_deviation, dev)
            details.append(f"Cluster {cluster_id}: dev={dev:.2f}")
        
        # 检查意外的簇
        for cluster_id in actual:
            if cluster_id not in profile:
                total_deviation += 0.5
                max_deviation = max(max_deviation, 0.5)
                details.append(f"Cluster {cluster_id}: unexpected")
        
        avg_deviation = total_deviation / max(1, len(profile))
        satisfied = avg_deviation <= tolerance
        value = 1.0 - min(1.0, avg_deviation)
        
        return {
            "satisfied": satisfied,
            "value": value,
            "detail": f"Profile dev={avg_deviation:.2f} (tol={tolerance}); " + "; ".join(details),
        }
    
    def _evaluate_spatial_pattern(self, prediction: StructuredPrediction, rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估空间模式：对称性、对齐、相对位置
        不依赖簇 ID，而是基于几何属性
        """
        pattern_type = rule.get("pattern_type", "symmetric")
        cluster_ids = rule.get("cluster_ids", [])  # 空则检查所有簇
        tolerance = rule.get("tolerance", 0.1)
        
        # 筛选区域
        if cluster_ids:
            regions = [r for r in prediction.regions if r.cluster_id in cluster_ids]
        else:
            regions = prediction.regions
        
        if len(regions) < 2:
            return {"satisfied": False, "value": 0.0, "detail": "Need at least 2 regions"}
        
        if pattern_type == "symmetric":
            # 检查是否存在关于中心对称的区域对
            best_val = 0.0
            satisfied = False
            for i, r1 in enumerate(regions):
                for r2 in regions[i+1:]:
                    cx1, cy1 = r1.center_xy
                    cx2, cy2 = r2.center_xy
                    sym_cx1, sym_cy1 = 1 - cx1, 1 - cy1
                    sym_dist = math.hypot(sym_cx1 - cx2, sym_cy1 - cy2)
                    val = max(0.0, 1.0 - sym_dist / 0.1)
                    if sym_dist < 0.1:
                        satisfied = True
                    if val > best_val:
                        best_val = val
            return {
                "satisfied": satisfied,
                "value": best_val,
                "detail": f"Symmetric pattern: best_val={best_val:.2f}",
            }
        
        elif pattern_type == "aligned_horizontal":
            # 水平对齐：y 坐标一致
            ys = [r.center_xy[1] for r in regions]
            spread = max(ys) - min(ys)
            val = max(0.0, 1.0 - spread / 0.1)
            satisfied = spread < 0.1
            return {
                "satisfied": satisfied,
                "value": val,
                "detail": f"AlignedH: spread={spread:.3f}",
            }
        
        elif pattern_type == "aligned_vertical":
            xs = [r.center_xy[0] for r in regions]
            spread = max(xs) - min(xs)
            val = max(0.0, 1.0 - spread / 0.1)
            satisfied = spread < 0.1
            return {
                "satisfied": satisfied,
                "value": val,
                "detail": f"AlignedV: spread={spread:.3f}",
            }
        
        elif pattern_type == "grid":
            # 网格排列：规则的行列分布
            # 简化版：检查是否大致形成网格
            xs = sorted([r.center_xy[0] for r in regions])
            ys = sorted([r.center_xy[1] for r in regions])
            # 简单检查：是否有明显的行列结构
            x_gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
            y_gaps = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
            x_uniform = max(x_gaps) / (min(x_gaps) + 1e-6) if x_gaps else 1.0
            y_uniform = max(y_gaps) / (min(y_gaps) + 1e-6) if y_gaps else 1.0
            uniformity = 1.0 - min(1.0, (x_uniform + y_uniform) / 4)
            return {
                "satisfied": uniformity > 0.5,
                "value": uniformity,
                "detail": f"Grid pattern: uniformity={uniformity:.2f}",
            }
        
        return {"satisfied": False, "value": 0.0, "detail": f"Unknown pattern: {pattern_type}"}
    
    def _evaluate_anomaly_score(self, prediction: StructuredPrediction, rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算综合异常分数：基于与正常模式的偏离
        不需要预先定义规则，自动计算异常程度
        """
        # 这里可以集成预训练的正常模式统计
        # 简化版：基于簇分布的熵、空间分布均匀度等
        
        # 1. 簇分布熵
        total_regions = len(prediction.regions)
        if total_regions == 0:
            return {"satisfied": False, "value": 0.0, "detail": "No regions"}
        
        cluster_counts = list(prediction.candidate_region_counts.values())
        total = sum(cluster_counts)
        probs = [c / total for c in cluster_counts]
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)
        max_entropy = math.log(len(cluster_counts))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 2. 空间分布均匀度
        if len(prediction.regions) >= 2:
            xs = [r.center_xy[0] for r in prediction.regions]
            ys = [r.center_xy[1] for r in prediction.regions]
            # 计算最近邻距离的变异系数
            dists = []
            for i, r1 in enumerate(prediction.regions):
                for r2 in prediction.regions[i+1:]:
                    dist = math.hypot(r1.center_xy[0] - r2.center_xy[0], 
                                      r1.center_xy[1] - r2.center_xy[1])
                    dists.append(dist)
            if dists:
                cv_dist = np.std(dists) / (np.mean(dists) + 1e-8)
            else:
                cv_dist = 0
        else:
            cv_dist = 0
        
        # 综合异常分数（越大越异常）
        # 正常样本：熵适中、分布均匀
        # 异常：熵过低（单一簇主导）或过高（碎片化）、分布不均
        anomaly_score = 0.5 * (1 - norm_entropy) + 0.5 * min(1.0, cv_dist)
        
        # 阈值：正常应 < 0.3
        threshold = rule.get("threshold", 0.3)
        satisfied = anomaly_score < threshold
        
        return {
            "satisfied": satisfied,
            "value": 1.0 - anomaly_score,  # 越大越正常
            "detail": f"AnomalyScore={anomaly_score:.3f} (entropy={norm_entropy:.2f}, cv_dist={cv_dist:.2f})",
        }

    def evaluate(self, prediction: StructuredPrediction, rule: Dict[str, Any]) -> Dict[str, Any]:
        rule_type = rule.get("type", "").lower()
        
        # 新增规则类型
        if rule_type == "cluster_profile":
            return self._evaluate_cluster_profile(prediction, rule)
        elif rule_type == "spatial_pattern":
            return self._evaluate_spatial_pattern(prediction, rule)
        elif rule_type == "anomaly_score":
            return self._evaluate_anomaly_score(prediction, rule)
    
    def evaluate_all(
        self,
        prediction: StructuredPrediction,
        rules: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        decision_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """执行多条规则，返回加权异常分数"""
        results = []
        if weights is None:
            weights = [1.0] * len(rules)
        
        for rule, weight in zip(rules, weights):
            result = self.evaluate(prediction, rule)
            results.append({
                "rule": rule,
                "weight": weight,
                **result,
            })
        
        # 计算加权异常分数：规则不满足的程度加权求和
        total_weight = sum(weights)
        weighted_anomaly = sum(
            w * (1.0 - r["value"]) for w, r in zip(weights, results)
        ) / total_weight
        
        # 决策：异常分数越大越异常
        is_anomaly = weighted_anomaly > decision_threshold
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": weighted_anomaly,
            "decision_threshold": decision_threshold,
            "results": results,
            "summary": f"anomaly_score={weighted_anomaly:.3f} (threshold={decision_threshold})",
        }


# ===== 便捷函数 =====

def create_structured_perception(
    n_clusters: int = 6,
    min_region_patches: int = 3,
    device: str = "cuda",
) -> StructuredPerception:
    """创建结构化感知提取器的便捷函数"""
    return StructuredPerception(
        n_clusters=n_clusters,
        min_region_patches=min_region_patches,
        device=device,
    )


def load_inpformer_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """从 INP-Former 提取编码器特征 en[-1]"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        en, de, g_loss = model(images)
        return en[-1].detach()  # [B, 768, 28, 28]


# ===== 测试脚本 =====

if __name__ == "__main__":
    import torch
    
    # 简单测试
    print("Testing StructuredPerception V1...")
    
    # 模拟 support features [K, 768, 28, 28]
    K = 4
    support_feat = torch.randn(K, 768, 28, 28)
    
    # 模拟 query features [1, 768, 28, 28]
    query_feat = torch.randn(1, 768, 28, 28)
    
    # 模拟 RGB 图像
    rgb = np.random.randint(0, 255, (392, 392, 3), dtype=np.uint8)
    
    # 创建提取器
    sp = create_structured_perception(n_clusters=6, device="cpu")
    
    # 拟合 support
    sp.fit_support(support_feat)
    print("Support fitted.")
    
    # 预测
    pred = sp.predict(query_feat, rgb_image=rgb)
    print(f"Regions: {len(pred.regions)}")
    print(f"Counts: {pred.candidate_region_counts}")
    print(f"Relations: {len(pred.relations)}")
    
    # 测试规则执行
    executor = RuleExecutor()
    rules = [
        {"type": "Exists", "cluster_id": 0, "min_count": 1},
        {"type": "LeftOf", "subject_cluster": 0, "object_cluster": 1},
        {"type": "Near", "cluster1": 0, "cluster2": 1, "threshold": 0.15},
    ]
    result = executor.evaluate_all(pred.regions, rules)
    print(f"Rule results: {result['summary']}")
    
    # 序列化
    serial = sp.to_serializable(pred)
    print(f"JSON output keys: {serial.keys()}")
    print("Test passed!")