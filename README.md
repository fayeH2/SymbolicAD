# SymbolicAD V5.1 — static neuro-symbolic FSAD

Staged neuro-symbolic anomaly detection for MVTec LOCO.

Qwen is an **offline rule author only**. Deployment inference never loads an
LLM: CLIP extracts visual features and the frozen neuro-symbolic program
evaluates its compiled expressions. For every query image the pipeline produces:

- a CLIP visual few-shot baseline heat-map;
- a component-grounding scene graph (text + support-prototype responses);
- relation-atom truth values and responsibility maps;
- a logic heat-map produced from failed rules;
- a fused heat-map plus image-level scores;
- Image / Pixel AUROC, AP, F1-max and AUPRO for baseline, logic-only and final;
- per-image visualisations, raw maps and JSON reasoning traces.

## Pipeline

```text
OFFLINE COMPILATION (LLM allowed only in generate_rules.py)
2 normal supports ─► Qwen rule proposal ─┐
2 normal supports ─► CLIP patch features ├─► validated static expression tree
                                        └─► frozen prototypes/thresholds/masks

DEPLOYMENT INFERENCE (LLM forbidden)
query ─► CLIP patches ─┬─► visual-memory anomaly predicate
                       ├─► semantic components ─► static rule tree
                       └─► visual prototypes ─► static dynamic expressions/MoE
                                                   │
                                            responsibility map ─► fusion
```

## Stages implemented

- **Stage 1** — data loading, CLIP visual few-shot baseline, metrics, heat-maps,
  baseline-only mode, tracked manual rule JSON.
- **Stage 2** — `Exist` + `CountEq` / `CountInRange` with component maps and
  connected-component instances.
- **Stage 3** — `OneToOne`, `Inside`, `LeftOf`, `RightOf`, `Above`, `Below`,
  `CenterAligned`, `CountRatio`, `AttributeMatch`, Hungarian matching and
  responsibility back-projection.
- **Rule compiler** — deduplicates atoms, removes implied/redundant atoms,
  calibrates counts from normal support, adds missing count rules and builds a
  hierarchical `AND/OR` tree with an audit report.
- **Offline Qwen3-VL rule generation** — Qwen3-VL-8B-Thinking inspects the normal support
  images, emits a structured component/count/relation/attribute description, and
  a deterministic builder converts it into atoms.  No category-specific rules
  are hand-written. Results are cached, then compiled; inference does not import
  or call the Qwen generator.

## Commands

2-shot baseline:

```bash
python run_pipeline.py --mode baseline --category juice_bottle --shot 2
python run_pipeline.py --mode baseline --all --shot 2
```

Compile a 2-shot static program, then run LLM-free inference:

```bash
python scripts/generate_rules.py --category juice_bottle --shot 2
python scripts/compile_static_program.py --category juice_bottle --shot 2 --device cuda:0
python run_pipeline.py --mode baseline_logic --category juice_bottle --shot 2
```

The inference command fails closed if the compiled program is missing or does
not match the support seed, CLIP model, image size or support images. It never
regenerates rules or support memory.

Every run writes to a fresh timestamped directory.  If a run was executed with
`--no-visualizations --no-maps`, the images can be reconstructed later from the
saved predictions and traces without re-running CLIP:

```bash
python scripts/render_saved_results.py --run-dir results_2shot/<timestamp>_k2
python scripts/render_saved_results.py --run-dir results_2shot/<timestamp>_k2 --limit 20
```

Two categories can be run in parallel on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --mode baseline --category juice_bottle --shot 2 --device cuda:0 &
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py --mode baseline --category pushpins --shot 2 --device cuda:0 &
```

Other entry points:

```bash
python scripts/generate_rules.py --all
python scripts/compile_static_program.py --all
python scripts/run_inference.py --category juice_bottle --mode baseline_logic
python scripts/evaluate.py
```

## Rule bank format

Rule knowledge bases live in `configs/rules/{category}.json` and are cached in
`cache/rules/{category}.json`. Allowed atoms:

```text
Exist(Component)
CountEq(Component, K)
CountInRange(Component, low, high)
CountRatio(ComponentA, ComponentB, Ratio)
OneToOne(ComponentA, ComponentB)
Inside / LeftOf / RightOf / Above / Below / CenterAligned (ComponentA, ComponentB)
AttributeMatch(Component, attribute, value)
```

Components may declare `role` and `part_of`. Attribute alternatives sharing a
`params.group` are compiled into an `OR` over joint attribute conditions.

The logic tree only references atom ids:

```json
{"op": "AND", "children": ["R1", "R2", "R3"]}
```

Rules are produced offline by Qwen3-VL from the support images, then validated
and compiled. The local model is `Qwen3-VL-8B-Thinking` (multimodal, so it can
inspect the support images directly).  `configs/rules/*.json` remains only as an
explicit fallback (`qwen.allow_template_fallback=true`); it is not used for
normal runs.  The manual `juice_bottle` template has been removed.

The compiler conservatively calibrates counts from support statistics and writes
an immutable deployment package under
`cache/static_programs/{category}_k{shot}/` containing `rules.json`,
`support_memory.npz`, `symbolicdet.npz` and `manifest.json`.
Violation prompts are compile-time documentation only; inference never performs
text-prompt rescoring.

## Few-shot baseline (PromptAD / AnomalyCLIP-style)

The baseline is **CLIP few-shot**, not zero-shot and not INP-Former.  It is
trained-free and category-agnostic; only the normal support images and the
category name are used:

- **Dense multi-layer features.** Patch tokens are read from several ViT
  blocks (`model.feature_layers`, e.g. `[3, 8, 11]`) instead of only the final
  projected layer.  Middle blocks preserve local structure.
- **Local neighbourhood smoothing.** Query/support features are blended with a
  3x3 average (`local_radius`, `local_weight`) before matching.
- **Fixed ensemble under strict FSAD.** `model.feature_source=both` is fixed
  before compilation. Query-time feature-space selection was deleted.
- **Visual-only anomaly predicate.** The baseline uses normal-support CLIP
  visual features only. Generic normal/damaged text prompts were removed.
- **Support-only calibration.** The image score is a top-k statistic of the
  support-calibrated map; optional whitening (`model.whiten`) is available but
  disabled by default because it damaged pixel localisation on LOCO.
- **Generic foreground prior.** The background colour is estimated from the
  image border (not hard-coded to black), and the prior is soft so it cannot
  erase small defects.

```text
native ViT layers ─┐
projected layer   ─┴─► support patch bank ─► local cosine kNN ─► H_visual
```

## Rule grounding, support calibration and fusion

The offline-generated semantic rules are aligned to pixel space in three stages:

1. **Support-calibrated atom truth.**  Each atom is evaluated with a
   leave-one-out support memory.  Atoms whose normal-support truth is low, or
   which fail on more than half of the held-out normal images, are disabled as
   grounding failures.  Query truths are expressed relative to the support
   q05/q50 range.
2. **Soft responsibility maps.**  Failed count/attribute/position/relation atoms
   project instance masks weighted by violation severity, not hard binary
   masks.  The logic map is smoothed and normalised by the leave-one-out normal
   logic noise floor.
3. **Gated fusion.** Pixel fusion is only enabled when the compiled support logic map is
   quiet (`fusion.max_support_logic_noise`); otherwise the symbolic branch
   abstains. Query-set reliability adaptation was deleted. Image-level symbolic
   evidence is added only from compiled support-calibrated weights.

```text
CLIP patches ─┬─► support patch bank ──────────────► H_base
              └─► component maps ─► instances ─► atoms ─► H_logic
H_final = 1 - (1 - H_base) * (1 - lambda * H_logic)
score_final = score_base + beta * score_logic
```

Rules are generated offline by Qwen3-VL and compiled into a static expression
tree. The same LLM-free executor runs for all five MVTec LOCO classes.

## V5 segment-first SymbolicDet + sparse MoE

The mentor-proposed supplement is implemented in `symbolicAD/symbolicdet` and
enabled by default in `baseline_logic` mode. During offline compilation it
clusters normal CLIP patch features into visual component prototypes, extracts
colour/spatial attributes from those masks, and fits explicit dynamic
interval expressions and evaluates appearance, structure and pair-relation
experts.  A per-image sparse router combines the strongest experts.  Missing
components are localised with their expected support masks, while relation
violations assign responsibility to both participating masks.

At inference the frozen CLIP prototypes and expressions are only evaluated; no
fitting occurs. This branch is fused independently from the compiled semantic
rule tree. Use `--no-symbolicdet` for the semantic-rule-only ablation and
`--no-semantic-rules` for the SymbolicDet-only ablation. Every run copies the
human-readable static expressions to
`checkpoint/<category>_symbolicdet.json`.  The detailed audit and required
ablation matrix are in `docs/v5_audit.md`.

## Historical 2-shot results (MVTec LOCO, 448 input)

The table below predates the strict support-only protocol guard and must not be
used as the final V5 comparison.  New runs record `protocol`, sample limits and
the effective symbolic fusion weights in every metrics JSON.

Baseline comparison before/after the dense-feature redesign:

| category | image AUROC old → new | pixel AUROC old → new | pixel AP old → new |
|---|---:|---:|---:|
| juice_bottle | 0.743 → **0.837** | 0.880 → **0.914** | 0.307 → **0.504** |
| breakfast_box | 0.713 → **0.822** | 0.863 → **0.903** | 0.356 → **0.600** |
| pushpins | 0.584 → **0.602** | 0.637 → **0.687** | 0.050 → **0.076** |
| screw_bag | 0.614 → 0.591 | 0.639 → **0.684** | 0.104 → **0.114** |
| splicing_connectors | 0.550 → **0.575** | 0.683 → **0.721** | 0.168 → **0.209** |

With symbolic fusion enabled, `juice_bottle` reaches image AUROC **0.847**.
The rule branch abstains automatically when its support/query calibration is
unreliable, so it cannot damage the baseline.

## Outputs

Every run creates a timestamped directory under the configured results root
(e.g. `1103results/20261104_140645_k8/`):

```text
{results_root}/{timestamp}_k{shot}/
├── csv/
│   ├── results.csv                 # H2VLR-style per-class i_roc / p_roc / ...
│   ├── metrics_by_variant.csv      # baseline / logic / final, all groups
│   └── metrics_per_class.csv       # detailed per class/group/variant
├── imgs/
│   ├── {category}-{defect}-{sample}_ori.jpg
│   ├── {category}-{defect}-{sample}_gt.jpg
│   └── {category}-{defect}-{sample}_{method}.jpg
├── checkpoint/
│   ├── {category}_rules.json
│   ├── {category}_rules.description.json
│   ├── {category}_rules.raw.txt
│   ├── {category}_support_memory_k{shot}.npz
│   ├── run_config.json
│   └── metrics_summary.json
├── maps/{category}/{group}_{sample}_{baseline,logic,final}.npy
├── traces/{category}/{group}_{sample}.json
├── predictions/{category}.npz
└── visualizations/{category}/{group}_{sample}.png
```

The three images per sample match the reference layout: original, GT overlay
(`_gt.jpg`, red mask), and prediction overlay (`_{method}.jpg`, jet heat-map).
`visualizations/{category}/{group}_{sample}.png` is a six-panel figure with
image / GT / baseline / logic / final / final overlay.  A quick
`visualizations/{category}/_contact_sheet.jpg` is also generated for the final
runs.
`results.csv` uses percent values and the same `i_roc, p_roc` naming as the
reference `Seed_111-results.csv`.

## Configuration

Numeric constants are centralised in `configs/default.yaml`. The most important
knobs are `model.feature_layers`, `model.feature_source`, `model.projected_weight`,
`logic.min_atom_support_truth`, `logic.support_calibration`,
`fusion.max_support_logic_noise`, `fusion.image_logic_weight`,
`evaluation.pixel_eval_size` and `output.*`.

## Tests

```bash
python -m pytest tests -q
```
