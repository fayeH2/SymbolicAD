# Qwen-guided relational normality rules for FSAD

> **Maintained LOCO path (2026-10-20):** use
> [`CORE_LOCO_PIPELINE.md`](CORE_LOCO_PIPELINE.md). The V2--V13 sections below
> are historical experiment records, not cumulative modules to enable together.
> A component enters the maintained path only after a paired ablation beats the
> baseline/simple control without test-label model selection.

This repository is an isolated neuro-symbolic layer around INP-Former.  It does
not modify the baseline repository.  Local
`/home/hjh/AgentModels/Qwen3-VL-8B-Thinking` is used once, offline, to induce a
normal scene program from few-shot normal images.  Test-time inference loads
only the compiled JSON rule library and deterministic visual grounder.

## V11 category-agnostic orthogonal feature fusion

Historical only: its non-monotone `NOT/IMPLIES/XOR` normality search has been
invalidated because simultaneous violations can cancel. The maintained search
accepts positive `AND/OR` normality trees only.

The newest research entry point is `run_generic_orthogonal_fsad.py`. It has no
category-specific algorithm branches. Qwen supplies a semantic entity seed;
the closed symbolic DSL enumerates executable relation candidates, validates
them by support-only cross-fitting and counterfactual intervention, removes
same-family/correlated rules, and evolves an `AND/OR/NOT/IMPLIES/XOR` AST. The
resulting logic map modulates individual INP encoder/decoder cosine-residual
layers before layer aggregation.

Fresh local-Qwen 4-shot transistor improves pixel AUPR from `0.30660` to
`0.32928`; VisA/candle improves all four metrics slightly; MVTec/screw is an
important zero-gain failure because the generic component grounder binds the
whole screw rather than the LLM's thread concept. See
[`ORTHOGONAL_FEATURE_FUSION_V11_REPORT.md`](ORTHOGONAL_FEATURE_FUSION_V11_REPORT.md)
for formulas, exact results, artifacts, protocol boundaries, and the remaining
grounding bottleneck.

## v7.1 executable AND/OR/NOT logic

The current controlled implementation replaces weighted rule-score addition
with an executable product-fuzzy AST.  Qwen proposes typed candidates
independently on normal supports; a fixed leave-one-normal-out and bootstrap
compiler accepts only stable, groundable rules.  Every accepted atom appears
exactly once in the symbolic AST, and the same program produces image scores
and pixel maps.  A strictly nested `k4 ⊂ k8 ⊂ k16 ⊂ k30` MVTec transistor
experiment improves all four requested metrics at k30:

| metric | frozen INP-Former | executable logic | delta |
|---|---:|---:|---:|
| I-AUROC | 0.906250 | 0.942917 | +0.036667 |
| I-AUPR | 0.884767 | 0.912037 | +0.027270 |
| P-AUROC | 0.806132 | 0.816702 | +0.010570 |
| P-AUPR | 0.306600 | 0.348465 | +0.041864 |

Run the resumable two-GPU experiment with:

```bash
bash scripts/run_transistor_executable_logic_2gpu.sh
```

See [`EXECUTABLE_LOGIC_EXPERIMENT_REPORT.md`](EXECUTABLE_LOGIC_EXPERIMENT_REPORT.md)
for the formal rules, nested-support negative controls, exact outputs, LLM-free
deployment command, and limitations.  This is a mechanism-development result
on one category, not yet an unseen-dataset generalization claim.

For the new all-GPU launcher covering 11 runnable industrial/medical datasets
and 103 categories, see
[`MULTIDATASET_TRAINING_GUIDE.md`](MULTIDATASET_TRAINING_GUIDE.md).  The short
entry point is `bash scripts/launch_all_datasets_2gpu.sh`; it is resumable and
keeps all new code/artifacts outside the INP-Former repository.

## Category-specific v5 multi-dataset experiment

The v5 branch tests the stronger hypothesis that each category needs its own
executable normality program.  It keeps INP-Former read-only and compiles Qwen
entities into separate grounders for MVTec `transistor`, VisA `pcb1` and
`macaroni2`, and MPDD `bracket_brown`.  With one frozen support-only fusion
formula, the first three datasets now all have a positive category result:

| dataset / class | INP-Former AUROC | + static rules | delta |
|---|---:|---:|---:|
| MVTec / transistor | 0.90625 | 0.92958 | +0.02333 |
| VisA / pcb1 | 0.95170 | 0.96960 | +0.01790 |
| MPDD / bracket_brown | 0.60106 | 0.62519 | +0.02413 |

VisA `macaroni2` is a useful negative control: its proposed four-piece count is
true for both normal and anomalous images, so all static rules abstain and the
score remains exactly `0.83910`.  The three positive-class macro delta is
`+0.02179`; including this negative control gives `+0.01634`.  Independent
recomputation and 2,000-resample bootstrap validation pass 40/40 checks.

These are exploratory category-development results, not a frozen 33-class
confirmation: the category grounders were designed after inspecting their
visual structure.  MVTec and VisA confidence intervals exclude zero; MPDD's
interval still crosses zero.  See
[`CLASS_SPECIFIC_EXPERIMENT_REPORT.md`](CLASS_SPECIFIC_EXPERIMENT_REPORT.md)
for failure analysis and exact artifacts.

v5 entry points:

- `run_class_specific_neurosymbolic_fsad.py`: compile and evaluate one category
  from cached INP-Former predictions.
- `run_class_specific_static_inference.py`: deploy a compiled library without
  loading Qwen.
- `scripts/validate_class_specific_v5.py`: independent metrics, bootstrap and
  leave-one-rule-out validation.
- `neurosymbolic_fsad/class_specific_rules.py`: isolated per-category scene
  programs and the static rule engine.

## Frozen v4 multi-dataset result

The class-agnostic repeated-component compiler has now been evaluated with one
frozen 4-shot protocol on all 15 MVTec AD, 12 VisA, and 6 MPDD classes.  Qwen
is queried independently on each of the four normal supports; at least 3/4
count agreement is required, and every executable rule must pass strict
leave-one-support-out grounder refitting with zero held-normal false positives.

Across 33 classes, 19 produced an LLM proposal, 13 had a stable grounder, and
only 6/89 candidate rules survived.  Macro image AUROC changed from `0.9459011`
to `0.9459476` (`+0.0000465`), with 2 improved classes and 0 degraded classes.
Independent recomputation passed 264/264 checks, but the nonzero gains are not
bootstrap-significant.  This is a safe result, not evidence of broad positive
generalization.

The diagnosis is important: the generic connected-component grounder does not
consume Qwen's entity description, so it can satisfy the requested count with
the wrong visual object.  The specialized semantic cable grounder remains the
positive proof of concept (`+0.04460` AUROC).  The complete protocol, per-dataset
table, failure analysis, and GroundingDINO/SAM-HQ remediation plan are in
[`MULTIDATASET_EXPERIMENT_REPORT.md`](MULTIDATASET_EXPERIMENT_REPORT.md).

v4 entry points:

- `scripts/generate_generic_relational_proposals.py`: independent local-Qwen
  proposals for a full benchmark.
- `scripts/run_generic_relational_suite.py`: frozen MVTec/VisA/MPDD suite.
- `run_generic_relational_inference.py`: compiled, LLM-free deployment.
- `scripts/validate_generic_relational_suite.py`: independent metric, lineage,
  consensus, and 2,000-resample bootstrap validation.
- `scripts/summarize_generic_relational_suite.py`: auditable cross-dataset
  coverage and metric summary.

## Why v3 is relational

The original v1/v2 prototypes allowed Qwen to select only global scalar or
anomaly-map features.  That cannot express rules such as “there are exactly
three conductors”, “one of each color identity must be present”, or “each
copper core is inside an insulator”.  The v3 flow is instead:

```text
4 known-normal images
        |
        +-- Qwen3-VL (GPU 1, offline) --> entities + executable predicates
        |
        +-- visual grounder ------------> support-derived part/color prototypes
                                            |
normal nuisances + relation counterfactuals-+--> verify/compile static JSON

test image --> INP-Former (GPU 0) + static grounder/rule engine --> score
                                                   (no LLM at runtime)
```

The executable DSL currently supports `count_eq`, `all_inside`,
`one_to_one_inside`, `distinct_attribute_count_eq`,
`attribute_multiset_eq`, `triangular_layout`, `similar_size`,
`centered_group`, and `relative_position`. Qwen does not choose thresholds or
fusion weights. Those are calibrated only from the four normal supports,
normal-preserving augmentations, and semantic counterfactuals. Every rule is
checked with strict leave-one-support-out CV: the visual grounder is refitted
from the other three supports before the held support is scored.

## First controlled result: MVTec cable, 4-shot

The first grounder is intentionally scoped to one class. It detects the inner
conductors, learns support-derived color identities, verifies the outer sheath
anchor, and executes two Qwen-proposed rules that survived support-only checks:
the exact color-identity multiset and a stable relative-position relation.

| metric | INP-Former | INP + relation | delta |
|---|---:|---:|---:|
| image AUROC | 0.94846 | **0.99307** | **+0.04460** |
| image AUPR | 0.97352 | **0.99486** | **+0.02134** |
| cable-swap AUROC | 0.61207 | **0.98563** | **+0.37356** |

The relation branch triggers all 12 cable-swap images and 1/58 normal test
images. Strict grounder-refit CV accepts color identity at AUROC `0.875`, FPR
`0.08333`, and counterfactual trigger rate `0.83333`; it accepts the retained
position rule at AUROC `0.875`, FPR `0`, and trigger rate `0.75`. It rejects the
count rule (CV AUROC `0.57292`), rejects one hallucinated position relation as
contradicted by normal support, and rejects geometry/containment where no
matching verifier exists.

An independent 2,000-resample paired bootstrap gives a 95% interval of
`[+0.01537, +0.08228]` for image AUROC and `[+0.00562, +0.04110]` for image
AUPR. The validator passes 48/48 lineage, metric, hash, cross-fit, fusion, and
LLM-free runtime checks.

This is an exploratory development result: the class was selected after
diagnosing the benchmark.  The training code never uses test labels, but a
confirmatory result requires freezing the method and evaluating a new class,
support seed, or dataset.

## Reproduce the dual-GPU experiment

```bash
cd /home/hjh/Desktop/NeuroSymbolic_FSAD
PYTHONDONTWRITEBYTECODE=1 conda run -n zeta \
  python run_relational_neurosymbolic_fsad.py \
  --dataset mvtec \
  --data-root /home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec \
  --class-name cable \
  --shot 4 \
  --support-strategy inp_official \
  --support-seed 1 \
  --checkpoint /home/hjh/Desktop/H2VLR_compare/INP-Former/saved_results/INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth \
  --inpformer-root /home/hjh/Desktop/H2VLR_compare/INP-Former \
  --llm-model /home/hjh/AgentModels/Qwen3-VL-8B-Thinking \
  --llm-model-hash c41fe4d3d3532dab697894040481b39a83ab7e53edc294d7c8c3561a91054b37 \
  --device cuda:0 \
  --llm-device cuda:1 \
  --allowed-rule-families identity,count,position,geometry,containment,core \
  --output-dir results_v3/mvtec/cable/k4_qwen_relational_strict
```

Use `--prediction-cache-dir results_v2_smoke/mvtec/cable/k4` to reuse exact
checkpoint/config-fingerprinted predictions. The v3 calibration cache also
records dataset/class, support seed and hashes, calibration seed, augmentation
schema, and every in-memory image digest. To replay a frozen Qwen proposal, add
`--proposal-json /absolute/path/to/qwen_relational_proposal.json`.

Important v3 outputs:

- `qwen_relational_proposal.json`: entities and candidate relation program.
- `candidate_audit_v3_relational.json`: support-only accept/reject evidence.
- `crossfit_manifest_v3_relational.json`: proof that every held support was
  excluded while fitting its fold's grounder.
- `normality_rules_v3_relational.json`: deployable static library.
- `predictions_v3_relational.csv`: baseline/rule/fused scores.
- `rule_evidence_v3_relational.json`: grounded entities and rule traces.
- `metrics_v3_relational.json`: overall and per-defect metrics.

## LLM-free static inference

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n zeta \
  python run_relational_static_inference.py \
  --rule-library results_v3/mvtec/cable/k4_strict_qwen_rules/normality_rules_v3_relational.json \
  --checkpoint /absolute/path/to/model.pth \
  --inpformer-root /home/hjh/Desktop/H2VLR_compare/INP-Former \
  --images /absolute/path/to/image1.png /absolute/path/to/image2.png \
  --output relational_predictions.json \
  --device cuda:0
```

The output explicitly records `runtime_loaded_llm: false` and verifies both the
checkpoint SHA-256 and all INP adapter settings against the compiled library.

## Code layout

- `neurosymbolic_fsad/qwen_relations.py`: strict relational JSON vocabulary and
  local Qwen3-VL proposer.
- `neurosymbolic_fsad/cable_relations.py`: cable scene grounding, semantic
  counterfactuals, rule compiler, and static engine.
- `neurosymbolic_fsad/experiment_relational.py`: controlled build/evaluation
  pipeline.
- `neurosymbolic_fsad/relational_inference.py`: deployment path without Qwen.
- `scripts/validate_relational_v3.py`: independent metric/leakage/bootstrap
  audit.

The earlier scalar v1/v2 runners remain available for ablation, but they are
not the intended relational method.

## v8: SymbolicDet-style evolutionary static rules

The controlled MVTec `transistor` path now implements the missing rule-search
stage from SymbolicDet. It evolves executable product-fuzzy ASTs with
population selection, crossover (`0.5`), mutation (`0.3`), complexity/FPR
penalties, and local Qwen3-VL contextual feedback. Search data contain only
known-normal support truths and auditable atom-space counterfactuals; query
images and labels are opened only after the winning AST is frozen.

Run the conservative two-atom and full four-atom searches concurrently:

```bash
cd /home/hjh/Desktop/NeuroSymbolic_FSAD
bash scripts/run_symbolicdet_evolution_v8_2gpu.sh
```

Main code and artifacts:

- `neurosymbolic_fsad/symbolicdet_evolution.py`: typed GA, fitness, mutation,
  crossover and lineage.
- `neurosymbolic_fsad/qwen_evolution_advisor.py`: correct local
  `Qwen3VLForConditionalGeneration` feedback path and strict AST validation.
- `run_symbolicdet_evolution_fsad.py`: support-only search followed by frozen
  query inference, four metrics, traces and heatmaps.
- `scripts/validate_symbolicdet_evolution_v8.py`: independent exact metric,
  support/query leakage, AST-node and paired bootstrap audit.
- `SYMBOLICDET_EVOLUTION_V8_REPORT.md`: results, ablations and limitations.

The valid v8 results must not be confused with the deprecated
`run_evolutionary_experiment.py` outputs, which used test labels during search.

## v12: Qwen + GroundingDINO + INP feature reasoning

V12 is the isolated open-vocabulary implementation. Qwen3-VL builds a static
class rule proposal from normal supports, GroundingDINO turns its entity text
into boxes and relational facts, evolutionary search produces an executable
fuzzy AST, and rule-localized INP residual evidence gates a positive feature
adapter. There are no category-specific inference branches and the LLM is not
loaded for query inference.

See `OPEN_VOCAB_SYMBOLIC_FSAD_V12_REPORT.md` for the architecture, exact
transistor/cable four-metric results, audit caveats and artifact locations.

## v13: shared INP entity tokens and full MVTec evaluation

V13 closes part of the representation gap by pooling GroundingDINO entity
slots from INP's own encoder/decoder feature grids. Its permutation-invariant
within-query entity relation can rescue a missing-part anomaly without first
requiring a high scalar baseline response, while negative correction is scoped
to the exact violated rule region.

The frozen 4-shot experiment over all 15 MVTec classes improves all four macro
metrics without regressing a class, but only cable and transistor activate the
bridge; the other classes safely abstain. See `V13_MVTEC_ENTITY_BRIDGE_REPORT.md`
for exact metrics, coverage and failure analysis.
