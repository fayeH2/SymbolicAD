# SymbolicAD V5 audit and experiment protocol

## Why a high score did not imply logical anomaly recognition

The completed 2-shot runs show that the observation is correct:

| run | samples | baseline image/pixel AUROC | logic image/pixel AUROC | final image/pixel AUROC | effective pixel logic weight |
|---|---:|---:|---:|---:|---:|
| juice_bottle `20261104_172359_576028_k2` | 330 | .8372 / .9139 | .5516 / .5005 | .8473 / .9139 | 0 |
| breakfast_box `20261104_172737_267669_k2` | 275 | .8234 / .9044 | .5191 / .5500 | .8234 / .9044 | 0 |

The final pixel AUROC was therefore the CLIP baseline value.  The old Qwen
logic branch was nearly random and its support-noise gate set `lambda_logic=0`.
The small juice-bottle image gain came from a 0.02 image-score residual, not
from improved logical localisation.  Three-sample smoke runs with AUROC 1.0
must not be compared with full-category results.

## Bugs and protocol risks removed

1. `model.feature_source=auto` and its test-query Otsu selector were deleted.
   The CLIP feature source is fixed before static-program compilation.
2. Query-side atom reliability disabled a rule when it was violated by most
   test queries.  That can erase the desired anomaly signal.  Strict evaluation
   The complete query-frequency path was deleted; rule reliability is frozen
   from the two normal support images.
3. Every metrics file now records the protocol, query adaptation, sample limit,
   effective fusion weights, whether logic abstained, and baseline-to-final
   AUROC deltas.
4. Qwen attributes with no support grounding are disabled instead of being
   treated as executable evidence.
5. Every run uses a fresh microsecond timestamped output directory.

There is no transductive mode in the deployment pipeline.

## Segment-first SymbolicDet implementation

The V5 supplement follows this frozen support-to-query flow:

```text
OFFLINE (Qwen allowed only in generate_rules.py)
2 normal supports -> Qwen semantic proposal
2 normal supports -> CLIP patch extraction
  -> CLIP visual prototypes and component masks
  -> masks and attributes (area/count/centroid/occupancy/Lab/edge)
  -> pair relations (dx/dy/distance)
  -> support-fitted dynamic interval expressions
  -> appearance / structure / relation experts
  -> per-image sparse MoE routing
  -> localised symbolic map

  -> compiled immutable static program

INFERENCE (no LLM imports/calls)
query -> CLIP patch features -> static semantic expression tree --\
                           -> static SymbolicDet expressions -------+-> fusion
                           -> visual-memory anomaly predicate ------/
```

The dynamic expressions use min/max support envelopes plus feature-specific
tolerance floors.  Their confidence is a deterministic function of two-shot
support stability.  Expected support masks let a missing component paint the
place where it should have appeared.  Relation violations paint both involved
components.  The sparse router chooses the strongest experts within a single
query, but never uses other query images, query labels, or test-set metrics.

Segmentation identities come from CLIP patch prototypes; Lab colour is an
explicit attribute calculated on those masks. The visual prototypes, expected
masks, expert priors and expressions are frozen under
`cache/static_programs/<category>_k<shot>/`. Inference fails if that package is
missing or incompatible and never rebuilds it.

## Required experiment matrix

Run every row with the same support seed and full test set (`max_images=0`):

| row | CLIP visual memory | compiled semantic rules | SegmentFirst | purpose |
|---|---:|---:|---:|---|
| B0 | yes | no | no | strict baseline |
| B1 | yes | yes | no | old symbolic contribution |
| B2 | yes | no | yes | mentor module contribution |
| V5 | yes | yes | yes | complete method |

CLI mapping:

```bash
# B0
python run_pipeline.py --mode baseline --all --shot 2
# B1
python run_pipeline.py --mode baseline_logic --all --shot 2 --no-symbolicdet
# B2
python run_pipeline.py --mode baseline_logic --all --shot 2 --no-semantic-rules
# V5
python run_pipeline.py --mode baseline_logic --all --shot 2
```

Report image AUROC separately for logical and structural groups, pixel AUROC/AP
and AUPRO, plus `logic_contribution` and qualitative failure cases.  The 94%
target is a goal, not a property to force through query-set selection; it is
accepted only when reached in full, strict, support-only runs across seeds.
