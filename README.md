# SymbolicAD V3.2

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
- **Stage 2** — `Exist` + `CountEq` with component maps and connected-component
  instances.
- **Stage 3** — `OneToOne`, `Inside`, `LeftOf`, `RightOf`, `Above`, `Below`,
  `CenterAligned`, `CountRatio`, Hungarian matching and responsibility
  back-projection.
- **Stage 4 (hook only)** — `symbolicAD/rules/qwen_rule_generator.py` keeps the
  cache/validation interface; Qwen3-VL generation is not called yet.

## Commands

Baseline-only smoke run (Stage 1 completion criterion):

```bash
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
CountRatio(ComponentA, ComponentB, Ratio)
OneToOne(ComponentA, ComponentB)
Inside / LeftOf / RightOf / Above / Below / CenterAligned (ComponentA, ComponentB)
```

The logic tree only references atom ids:

```json
{"op": "AND", "children": ["R1", "R2", "R3"]}
```

Invalid atoms are disabled (weight 0) instead of terminating the run. Count and
ratio atoms that contradict normal support statistics are also disabled.

## Outputs

```text
results/metrics_per_class.csv
results/metrics_summary.json
results/visualizations/{category}/{group}_{image}.png
results/maps/{category}/{group}_{image}_{baseline,logic,final}.npy
results/traces/{category}/{group}_{image}.json
results/predictions/{category}.npz
cache/rules/{category}.json
cache/support_memory/{category}_k{shot}.npz
```

## Configuration

Numeric constants are centralised in `configs/default.yaml`. The most important
knobs are `grounding.alpha`, `logic.*`, `fusion.lambda_logic`, `fusion.beta`,
`evaluation.pixel_eval_size` and `output.*`.

## Tests

```bash
python -m pytest tests -q
```
