"""Integrated neuro-symbolic FSAD experiment runner."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from PIL import Image

from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter, load_prediction_cache, save_prediction_cache
from neurosymbolic_fsad.perception.segmentation import ComponentSegmenter, FallbackSegmenter
from neurosymbolic_fsad.perception.scene_graph import (
    SceneGraph, build_scene_graph, align_scene_graphs, compute_normal_slots
)
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank, match_components_to_prototypes
from neurosymbolic_fsad.rules.generation.dsl import RuleSet, NormalityRule, RuleType
from neurosymbolic_fsad.rules.generation.qwen_interface import QwenVLRuleGenerator, QwenVLRuleRefiner
from neurosymbolic_fsad.rules.search.nsga2_search import run_rule_search, SearchConfig
from neurosymbolic_fsad.reasoning.executor import RuleExecutor, RuleExecutionResult
from neurosymbolic_fsad.grounding.grounder import RuleToPixelGrounder, GroundingFusion, create_logic_anomaly_map
from neurosymbolic_fsad.fusion.fusion import AnomalyFusion, FusionConfig, compute_grounding_confidence
from neurosymbolic_fsad.loco import (
    get_loco_data_root, select_loco_support, collect_loco_test_samples,
    LOCOEvaluator, LOCO_CLASSES
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IntegratedExperiment:
    """Full neuro-symbolic FSAD pipeline."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.data_root = Path(args.data_root).expanduser().resolve()
        self.dataset = args.dataset
        self.class_name = args.class_name
        self.shot = args.shot
        self.seed = args.seed
        
        # INP-Former
        self.checkpoint = Path(args.checkpoint).expanduser().resolve()
        self.inpformer_root = Path(args.inpformer_root).expanduser().resolve()
        self.encoder = args.encoder
        self.inp_num = args.inp_num
        self.input_size = args.input_size
        self.crop_size = args.crop_size
        self.map_size = args.map_size
        
        # Initialize adapter
        self.adapter = INPFormerAdapter(
            checkpoint=self.checkpoint,
            device=args.device,
            inpformer_root=self.inpformer_root,
            encoder_name=self.encoder,
            inp_num=self.inp_num,
            input_size=self.input_size,
            crop_size=self.crop_size,
            map_size=self.map_size,
        )
        
        # Perception
        self.segmenter = self._create_segmenter()
        
        # Fusion
        self.fusion = AnomalyFusion(FusionConfig(
            visual_weight=args.visual_weight,
            logic_weight=args.logic_weight,
            fusion_method=args.fusion_method,
        ))
        
        # Grounding
        self.grounder = RuleToPixelGrounder(map_size=self.map_size)
        self.grounding_fusion = GroundingFusion(map_size=self.map_size)
        
        # State
        self.rule_set: Optional[RuleSet] = None
        self.prototype_bank: Optional[SemanticPrototypeBank] = None
        self.normal_slots: Dict[int, Dict[str, Any]] = {}
        self.support_graphs: List[SceneGraph] = []
    
    def _create_segmenter(self) -> ComponentSegmenter:
        """Create component segmenter."""
        if self.args.sam_checkpoint and Path(self.args.sam_checkpoint).exists():
            return ComponentSegmenter(
                sam_checkpoint=self.args.sam_checkpoint,
                model_type=self.args.sam_model_type,
                device=self.args.device,
            )
        else:
            print("SAM checkpoint not found, using fallback segmenter")
            return FallbackSegmenter()
    
    def run_offline_phase(self) -> Dict[str, Any]:
        """Run offline phase: rule generation, search, and memory construction."""
        print("=" * 60)
        print("OFFLINE PHASE: Rule Generation & Memory Construction")
        print("=" * 60)
        
        # 1. Get support images
        if self.dataset == "mvtec_loco":
            support_paths = select_loco_support(
                self.data_root, self.class_name, self.shot, self.seed
            )
        else:
            support_paths = select_inpformer_support(
                self.data_root, self.dataset, self.class_name, self.shot, self.seed
            )
        
        print(f"Selected {len(support_paths)} support images")
        
        # 2. Extract INP-Former features for supports
        support_ids = [str(p.resolve()) for p in support_paths]
        support_preds = self.adapter.infer_images(
            support_paths, batch_size=self.args.batch_size,
            identifiers=support_ids, return_layer_maps=True
        )
        
        # 3. Segment components and build scene graphs
        print("Building scene graphs...")
        self.support_graphs = []
        support_masks = []
        
        for path in support_paths:
            components = self.segmenter.segment(path, return_features=True)
            graph = build_scene_graph(components)
            self.support_graphs.append(graph)
            
            # Combine component masks for prototype building
            combined_mask = np.zeros((self.map_size, self.map_size), dtype=bool)
            for comp in components:
                mask_resized = cv2.resize(
                    comp.mask.astype(np.float32),
                    (self.map_size, self.map_size),
                ) > 0.5
                combined_mask = combined_mask | mask_resized
            support_masks.append(combined_mask)
        
        # 4. Align support graphs and compute normal slots
        print("Aligning support graphs...")
        matches_list = []
        for i in range(1, len(self.support_graphs)):
            matches, _ = align_scene_graphs(self.support_graphs[0], self.support_graphs[i])
            matches_list.append(matches)
        
        self.normal_slots = compute_normal_slots(self.support_graphs, matches_list)
        print(f"Computed {len(self.normal_slots)} normal slots")
        
        # 5. Build prototype bank
        print("Building prototype bank...")
        self.prototype_bank = SemanticPrototypeBank()
        self.prototype_bank.build_from_support(self.support_graphs, matches_list)
        print(f"Built {len(self.prototype_bank.prototypes)} prototypes")
        
        # 6. Match support components to prototypes
        for i, graph in enumerate(self.support_graphs):
            matches = match_components_to_prototypes(
                graph.components, self.prototype_bank, threshold=0.4
            )
            for comp_idx, part_type in matches.items():
                graph.component_types[comp_idx] = part_type
        
        # 7. Generate rules with Qwen-VL (if enabled)
        qwen_rules = []
        if self.args.use_qwen and self.args.qwen_model:
            print("Generating rules with Qwen-VL...")
            qwen_gen = QwenVLRuleGenerator(
                model_path=self.args.qwen_model,
                device=self.args.device,
            )
            qwen_rules = qwen_gen.generate_rules(
                self.class_name, support_paths, self.shot
            )
            print(f"Qwen generated {len(qwen_rules)} candidate rules")
            
            # Refine rules
            refiner = QwenVLRuleRefiner(qwen_gen)
            qwen_rules = refiner.refine_rules(qwen_rules, support_paths, self.support_graphs)
            print(f"After refinement: {len(qwen_rules)} rules")
        
        # 8. Run NSGA-II rule search
        print("Running NSGA-II rule search...")
        search_config = SearchConfig(
            population_size=self.args.pop_size,
            max_generations=self.args.max_gen,
            random_seed=self.seed,
        )
        
        # Use Qwen rules as initial population, or generate defaults
        initial_rules = qwen_rules if qwen_rules else self._generate_default_rules()
        
        # Prepare normal masks for search evaluator
        normal_masks = [m.astype(np.float32) for m in support_masks]
        
        final_rule_set, search_history = run_rule_search(
            initial_rules, self.support_graphs, self.prototype_bank,
            normal_masks, search_config
        )
        
        self.rule_set = final_rule_set
        print(f"Final rule set: {len(self.rule_set.rules)} rules")
        
        # Save artifacts
        self.rule_set.save(self.output_dir / "normality_rules.json")
        self.prototype_bank.save(self.output_dir / "prototypes.json")
        _dump(self.output_dir / "normal_slots.json", self.normal_slots)
        _dump(self.output_dir / "search_history.json", search_history)
        
        return {
            "support_paths": [str(p) for p in support_paths],
            "num_rules": len(self.rule_set.rules),
            "num_prototypes": len(self.prototype_bank.prototypes),
            "num_slots": len(self.normal_slots),
            "search_history": search_history,
        }
    
    def _generate_default_rules(self) -> List[NormalityRule]:
        """Generate basic default rules from support statistics."""
        rules = []
        rule_id = 0
        
        # Count rules for each part type
        for part_type, proto in self.prototype_bank.prototypes.items():
            rule = NormalityRule(
                rule_id=f"{self.class_name}_count_{part_type}",
                rule_type=RuleType.COUNT,
                description=f"Expected count of {part_type}",
                part_type=part_type,
                expected_count=proto.count,
                count_tolerance=1,
                confidence=0.8,
                source="default",
            )
            rules.append(rule)
            rule_id += 1
        
        return rules
    
    def run_online_phase(self) -> Dict[str, Any]:
        """Run online phase: inference on test images."""
        print("=" * 60)
        print("ONLINE PHASE: Query Inference")
        print("=" * 60)
        
        # Load rule set if not already loaded
        if self.rule_set is None:
            rule_path = self.output_dir / "normality_rules.json"
            if rule_path.exists():
                self.rule_set = RuleSet.load(rule_path)
            else:
                raise ValueError("Rule set not found. Run offline phase first.")
        
        # Load prototypes if not already loaded
        if self.prototype_bank is None:
            proto_path = self.output_dir / "prototypes.json"
            if proto_path.exists():
                self.prototype_bank = SemanticPrototypeBank.load(proto_path)
        
        # Load normal slots
        slots_path = self.output_dir / "normal_slots.json"
        if slots_path.exists():
            self.normal_slots = json.loads(slots_path.read_text())
            # Convert string keys back to int
            self.normal_slots = {int(k): v for k, v in self.normal_slots.items()}
        
        # Get test samples
        if self.dataset == "mvtec_loco":
            class_root = self.data_root / self.class_name
            samples = collect_loco_test_samples(class_root)
        else:
            class_root = self.data_root / self.class_name
            samples = collect_test_samples(class_root)
        
        test_paths = [s["path"] for s in samples]
        test_ids = [str(p.resolve()) for p in test_paths]
        
        # Get INP-Former predictions
        print("Running INP-Former inference...")
        cache_dir = Path(self.args.cache_dir).expanduser().resolve() if self.args.cache_dir else self.output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "schema": "inp-layer-residual-v4",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "dataset": self.dataset,
            "class_name": self.class_name,
            "shot": self.shot,
            "support_seed": self.seed,
            "encoder": self.encoder,
            "inp_num": self.inp_num,
            "input_size": self.input_size,
            "crop_size": self.crop_size,
            "map_size": self.map_size,
        }
        
        test_cache_path = cache_dir / "test_inp_v4.npz"
        test_predictions = load_prediction_cache(test_cache_path, test_ids, expected_metadata=metadata)
        
        if test_predictions is None:
            test_predictions = self.adapter.infer_images(
                test_paths, batch_size=self.args.batch_size,
                identifiers=test_ids, return_layer_maps=True
            )
            save_prediction_cache(test_cache_path, test_predictions, metadata)
            print("Saved prediction cache")
        else:
            print("Loaded prediction cache")
        
        # Run inference
        print("Running neuro-symbolic inference...")
        executor = RuleExecutor(self.rule_set, self.normal_slots, self.map_size)
        
        results = []
        all_visual_scores = []
        all_logic_scores = []
        all_fused_scores = []
        all_visual_maps = []
        all_logic_maps = []
        all_fused_maps = []
        all_gt_masks = []
        all_labels = []
        
        for idx, (sample, baseline_score, baseline_map) in enumerate(zip(
            samples,
            test_predictions["scores"],
            test_predictions["maps"],
        )):
            if idx % 20 == 0:
                print(f"  Processing {idx}/{len(samples)}")
            
            image_path = sample["path"]
            label = sample["label"]
            
            # Segment query
            query_components = self.segmenter.segment(image_path, return_features=True)
            query_graph = build_scene_graph(query_components)
            
            # Match to prototypes
            matches = match_components_to_prototypes(
                query_graph.components, self.prototype_bank, threshold=0.4
            )
            for comp_idx, part_type in matches.items():
                query_graph.component_types[comp_idx] = part_type
            
            # Execute rules
            execution_result = executor.execute(query_graph, str(image_path))
            
            # Ground violations to pixel space
            logic_map, contributions = create_logic_anomaly_map(
                execution_result, query_graph, self.normal_slots,
                self.grounder, self.grounding_fusion,
                rule_weights={r.rule_id: r.confidence for r in self.rule_set.rules}
            )
            
            # Compute grounding confidence
            grounding_conf = compute_grounding_confidence(logic_map, baseline_map)
            
            # Fuse with INP-Former
            fusion_result = self.fusion.fuse(
                visual_map=baseline_map,
                visual_score=baseline_score,
                logic_map=logic_map,
                logic_score=execution_result.overall_score,
                grounding_confidence=grounding_conf,
                rule_contributions=contributions,
            )
            
            # Get ground truth mask
            gt_mask = self._get_gt_mask(class_root, sample)
            
            # Store results
            all_visual_scores.append(baseline_score)
            all_logic_scores.append(execution_result.overall_score)
            all_fused_scores.append(fusion_result.fused_score)
            all_visual_maps.append(baseline_map)
            all_logic_maps.append(logic_map)
            all_fused_maps.append(fusion_result.fused_map)
            all_gt_masks.append(gt_mask)
            all_labels.append(label)
            
            results.append({
                "path": str(image_path),
                "defect_type": sample["defect_type"],
                "label": label,
                "visual_score": float(baseline_score),
                "logic_score": float(execution_result.overall_score),
                "fused_score": float(fusion_result.fused_score),
                "violated_rules": [v.rule_id for v in execution_result.violations],
                "num_violations": len(execution_result.violations),
            })
            
            # Save visualizations
            if idx < self.args.max_visualizations:
                self._save_visualization(
                    idx, image_path, gt_mask,
                    baseline_map, logic_map, fusion_result.fused_map,
                    execution_result, fusion_result,
                    sample["defect_type"]
                )
        
        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_labels),
            np.array(all_visual_scores),
            np.array(all_logic_scores),
            np.array(all_fused_scores),
            np.array(all_gt_masks),
            np.array(all_visual_maps),
            np.array(all_logic_maps),
            np.array(all_fused_maps),
        )
        
        # Save results
        _dump(self.output_dir / "predictions.json", results)
        _dump(self.output_dir / "metrics.json", metrics)
        
        np.savez_compressed(
            self.output_dir / "maps.npz",
            paths=np.array([str(p) for p in test_paths]),
            labels=np.array(all_labels),
            masks=np.array(all_gt_masks),
            visual_maps=np.array(all_visual_maps),
            logic_maps=np.array(all_logic_maps),
            fused_maps=np.array(all_fused_maps),
            visual_scores=np.array(all_visual_scores),
            logic_scores=np.array(all_logic_scores),
            fused_scores=np.array(all_fused_scores),
        )
        
        return metrics
    
    def _get_gt_mask(self, class_root: Path, sample: Dict[str, Any]) -> np.ndarray:
        """Get ground truth mask."""
        if sample["label"] == 0:
            return np.zeros((self.map_size, self.map_size), dtype=bool)
        
        image_path = Path(sample["path"])
        defect = image_path.parent.name
        
        candidates = [
            class_root / "ground_truth" / defect / f"{image_path.stem}_mask.png",
            class_root / "ground_truth" / defect / f"{image_path.stem}.png",
            class_root / "test" / "ground_truth" / defect / f"{image_path.stem}_mask.png",
        ]
        
        for c in candidates:
            if c.exists():
                with Image.open(c) as img:
                    mask = np.array(img.convert("L"))
                import cv2
                mask = cv2.resize(
                    mask, (self.map_size, self.map_size),
                    interpolation=cv2.INTER_NEAREST
                )
                return mask > 128
        
        return np.zeros((self.map_size, self.map_size), dtype=bool)
    
    def _compute_metrics(
        self,
        labels: np.ndarray,
        visual_scores: np.ndarray,
        logic_scores: np.ndarray,
        fused_scores: np.ndarray,
        masks: np.ndarray,
        visual_maps: np.ndarray,
        logic_maps: np.ndarray,
        fused_maps: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute AUROC/AUPR metrics."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        def compute_pair(lbls, scrs, prefix):
            return {
                f"{prefix}_auroc": float(roc_auc_score(lbls, scrs)),
                f"{prefix}_ap": float(average_precision_score(lbls, scrs)),
            }
        
        # Image-level
        metrics = {
            **compute_pair(labels, visual_scores, "visual"),
            **compute_pair(labels, logic_scores, "logic"),
            **compute_pair(labels, fused_scores, "fused"),
        }
        
        # Pixel-level
        pixel_labels = masks.reshape(len(masks), -1)
        metrics.update(compute_pair(
            pixel_labels.flatten(), visual_maps.reshape(len(visual_maps), -1).flatten(), "visual_pix"
        ))
        metrics.update(compute_pair(
            pixel_labels.flatten(), logic_maps.reshape(len(logic_maps), -1).flatten(), "logic_pix"
        ))
        metrics.update(compute_pair(
            pixel_labels.flatten(), fused_maps.reshape(len(fused_maps), -1).flatten(), "fused_pix"
        ))
        
        # Deltas
        for key in ["auroc", "ap"]:
            v_key = f"visual_{key}"
            f_key = f"fused_{key}"
            if v_key in metrics and f_key in metrics:
                metrics[f"delta_fused_vs_visual_{key}"] = metrics[f_key] - metrics[v_key]
        
        return metrics
    
    def _save_visualization(
        self, idx: int, image_path: Path, gt_mask: np.ndarray,
        visual_map: np.ndarray, logic_map: np.ndarray, fused_map: np.ndarray,
        execution_result: RuleExecutionResult, fusion_result: FusionResult,
        defect_type: str,
    ) -> None:
        """Save comparison visualization."""
        import cv2
        
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess original image
        with Image.open(image_path) as img:
            original = np.array(img.convert("RGB"))
        
        # Resize to map size
        original = cv2.resize(original, (self.map_size, self.map_size))
        
        def overlay(img, map_):
            map_uint8 = (np.clip(map_, 0, 1) * 255).astype(np.uint8)
            heat = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(img, 0.5, heat, 0.5, 0)
        
        def add_label(img, text):
            result = img.copy()
            cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(result, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 1, cv2.LINE_AA)
            return result
        
        # GT overlay
        gt_vis = original.copy()
        if np.any(gt_mask):
            contours = cv2.findContours(
                gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            cv2.drawContours(gt_vis, contours, -1, (255, 0, 0), 2)
        
        panels = [
            add_label(original, "Original"),
            add_label(gt_vis, f"GT ({defect_type})"),
            add_label(overlay(original, visual_map), f"INP-Former ({visual_map.max():.3f})"),
            add_label(overlay(original, logic_map), f"Logic ({logic_map.max():.3f})"),
            add_label(overlay(original, fused_map), f"Fused ({fused_map.max():.3f})"),
        ]
        
        comparison = np.concatenate(panels, axis=1)
        Image.fromarray(comparison).save(
            vis_dir / f"{idx:04d}_{image_path.parent.name}_{image_path.stem}.jpg",
            quality=95
        )
    
    def run(self) -> Dict[str, Any]:
        """Run full experiment."""
        print(f"Starting integrated experiment: {self.dataset}/{self.class_name} K={self.shot}")
        
        # Offline phase
        offline_results = self.run_offline_phase()
        
        # Online phase
        online_results = self.run_online_phase()
        
        # Combine results
        final_results = {
            "dataset": self.dataset,
            "class_name": self.class_name,
            "shot": self.shot,
            "seed": self.seed,
            "offline": offline_results,
            "online": online_results,
            "config": vars(self.args),
        }
        
        _dump(self.output_dir / "final_results.json", final_results)
        
        print("=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Visual AUROC: {online_results.get('visual_auroc', 0):.4f}")
        print(f"Logic AUROC: {online_results.get('logic_auroc', 0):.4f}")
        print(f"Fused AUROC: {online_results.get('fused_auroc', 0):.4f}")
        print(f"Results saved to: {self.output_dir}")
        
        return final_results
    
    def close(self):
        """Clean up resources."""
        self.adapter.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated Neuro-Symbolic FSAD")
    
    # Data
    parser.add_argument("--dataset", type=str, default="mvtec",
                       choices=["mvtec", "visa", "mvtec_loco"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--class-name", type=str, default="transistor")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=668)
    
    # INP-Former
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--inpformer-root", type=str,
                       default="/home/hjh/Desktop/H2VLR_compare/INP-Former")
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    
    # Qwen-VL
    parser.add_argument("--use-qwen", action="store_true")
    parser.add_argument("--qwen-model", type=str,
                       default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    
    # Rule search
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--max-gen", type=int, default=30)
    
    # Fusion
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--logic-weight", type=float, default=1.0)
    parser.add_argument("--fusion-method", type=str, default="grounded_and_or",
                       choices=["weighted_sum", "probabilistic_or", "grounded_and_or"])
    
    # SAM
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_h")
    
    # Runtime
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-visualizations", type=int, default=50)
    
    return parser


def main():
    args = build_parser().parse_args()
    exp = IntegratedExperiment(args)
    try:
        results = exp.run()
        print(json.dumps(results, indent=2, default=_json_default))
    finally:
        exp.close()


if __name__ == "__main__":
    main()