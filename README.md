# SymbolicAD V4.0

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

Baseline-only smoke run (Stage 1 completion criterion):

```bash
python scripts/generate_rules.py --category juice_bottle --shot 8
python run_pipeline.py --mode baseline --category juice_bottle --max-images 20
```

Baseline + logic (Stages 2–3):

```bash
python run_pipeline.py --mode baseline_logic --category juice_bottle
python run_pipeline.py --mode baseline_logic --all
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

## Baseline and semantic-visual fusion

The baseline is **CLIP few-shot**, not zero-shot and not INP-Former.  It builds
a normal patch bank from the same support images that Qwen uses for rule
generation, and it uses the same `CLIPBackbone` patch features as component
grounding:

```text
CLIP patches ──► normal support patch bank ──► baseline heat-map H_base
             └─► component maps / atom responsibilities ──► logic heat-map H_logic
H_final = 1 - (1 - H_base) * (1 - lambda * H_logic)
```

Thus the symbolic rules and the baseline live in the same CLIP visual-semantic
space and are fused on the same patch grid.  A pure visual baseline such as
INP-Former would require an additional projection/alignment mechanism before
this fusion could be meaningful.

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
`results.csv` uses percent values and the same `i_roc, p_roc` naming as the
reference `Seed_111-results.csv`.

## Configuration

Numeric constants are centralised in `configs/default.yaml`. The most important
knobs are `grounding.alpha`, `logic.*`, `fusion.lambda_logic`, `fusion.beta`,
`evaluation.pixel_eval_size` and `output.*`.

## Tests

```bash
python -m pytest tests -q
```
