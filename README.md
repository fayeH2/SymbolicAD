# SymbolicAD V4.1

Staged neuro-symbolic anomaly detection for MVTec LOCO.

The first version is intentionally built to **run end-to-end first, then verify
and replace individual modules**. For every query image the pipeline produces:

- a CLIP zero-shot baseline heat-map;
- a component-grounding scene graph (text + support-prototype responses);
- relation-atom truth values and responsibility maps;
- a logic heat-map produced from failed rules;
- a fused heat-map plus image-level scores;
- Image / Pixel AUROC, AP, F1-max and AUPRO for baseline, logic-only and final;
- per-image visualisations, raw maps and JSON reasoning traces.

## Pipeline

```text
support images ──► CLIP support memory (prototypes, slots, relation statistics)
      │                         │
      └──► rule bank JSON ◄─────┘
                    │
query image ─► CLIP patches ─┬─► zero-shot baseline map
                             └─► component maps ─► instances ─► scene graph
                                                        │
                                               relation atoms ─► logic tree
                                                        │
                                        responsibility maps + CLIP rescore
                                                        │
                                               logic map ─► fusion ─► metrics
```

## Stages implemented

- **Stage 1** — data loading, CLIP zero-shot baseline, metrics, heat-maps,
  baseline-only mode, tracked manual rule JSON.
- **Stage 2** — `Exist` + `CountEq` / `CountInRange` with component maps and
  connected-component instances.
- **Stage 3** — `OneToOne`, `Inside`, `LeftOf`, `RightOf`, `Above`, `Below`,
  `CenterAligned`, `CountRatio`, `AttributeMatch`, Hungarian matching and
  responsibility back-projection.
- **Rule compiler** — deduplicates atoms, removes implied/redundant atoms,
  calibrates counts from normal support, adds missing count rules and builds a
  hierarchical `AND/OR` tree with an audit report.
- **Qwen3-VL rule generation** — Qwen3-VL-8B-Thinking inspects the normal support
  images, emits a structured component/count/relation/attribute description, and
  a deterministic builder converts it into atoms.  No category-specific rules
  are hand-written.  Results are cached under `cache/qwen_rules/`.

## Commands

2-shot baseline:

```bash
python run_pipeline.py --mode baseline --category juice_bottle --shot 2
python run_pipeline.py --mode baseline --all --shot 2
```

2-shot baseline + logic (automatic Qwen rules and support calibration):

```bash
python run_pipeline.py --mode baseline_logic --category juice_bottle --shot 2
python run_pipeline.py --mode baseline_logic --all --shot 2
```

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
python scripts/build_support_memory.py --category juice_bottle
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

Rules are produced by Qwen3-VL from the support images, then validated and
compiled.  The local model is `Qwen3-VL-8B-Thinking` (multimodal, so it can
inspect the support images directly).  `configs/rules/*.json` remains only as an
explicit fallback (`qwen.allow_template_fallback=true`); it is not used for
normal runs.  The manual `juice_bottle` template has been removed.

The compiler conservatively calibrates counts from support statistics (keeping
a rule value when support is noisy) and writes
`cache/rules/{category}.compiled.json` plus `.compile.json`.
`logic.rescore_enabled` defaults to `false`, so `violation_prompt` does not
affect heat-maps unless explicitly enabled.

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
  before seeing test queries.  The former Otsu `auto` selector remains only as
  an explicitly enabled transductive ablation (`--allow-transductive`) and its
  results are marked separately.
- **Textual normal-vs-damaged branch.** Generic normal/damaged prompts are
  calibrated on support and fused at image level; the pixel map uses the visual
  memory branch, because text maps were not spatially reliable in the 2-shot
  setting.
- **Support-only calibration.** The image score is a top-k statistic of the
  support-calibrated map; optional whitening (`model.whiten`) is available but
  disabled by default because it damaged pixel localisation on LOCO.
- **Generic foreground prior.** The background colour is estimated from the
  image border (not hard-coded to black), and the prior is soft so it cannot
  erase small defects.

```text
native ViT layers ─┐
projected layer   ─┼─► support patch bank ─► local cosine kNN ─► H_visual
                   │
CLIP text prompts ─┴─► normal/damaged logits ─► image-level residual
```

## Rule grounding, support calibration and fusion

The Qwen rules are aligned to pixel space in three stages:

1. **Support-calibrated atom truth.**  Each atom is evaluated with a
   leave-one-out support memory.  Atoms whose normal-support truth is low, or
   which fail on more than half of the held-out normal images, are disabled as
   grounding failures.  Query truths are expressed relative to the support
   q05/q50 range.
2. **Soft responsibility maps.**  Failed count/attribute/position/relation atoms
   project instance masks weighted by violation severity, not hard binary
   masks.  The logic map is smoothed and normalised by the leave-one-out normal
   logic noise floor.
3. **Gated fusion.**  Pixel fusion is only enabled when the support logic map is
   quiet (`fusion.max_support_logic_noise`); otherwise the symbolic branch
   abstains.  Strict evaluation never disables atoms from query-set statistics:
   doing so can erase a valid rule precisely because many anomalies violate it.
   The old query-frequency filter is available only in transductive ablations.
   Image-level symbolic evidence is added as a small support-calibrated residual.

```text
CLIP patches ─┬─► support patch bank ──────────────► H_base
              └─► component maps ─► instances ─► atoms ─► H_logic
H_final = 1 - (1 - H_base) * (1 - lambda * H_logic)
score_final = score_base + beta * score_logic
```

Rules are generated automatically by Qwen3-VL; there are no category-specific
constants, templates or hand-written atoms.  The same pipeline runs for all
five MVTec LOCO classes.

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
`grounding.alpha`, `logic.min_atom_support_truth`, `logic.support_calibration`,
`fusion.max_support_logic_noise`, `fusion.image_logic_weight`,
`evaluation.pixel_eval_size` and `output.*`.

## Tests

```bash
python -m pytest tests -q
```
