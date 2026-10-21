"""Documentation of INP-Former + DINOv2 Compatibility Issue.

## Root Cause

INP-Former's architecture assumes the encoder's intermediate features include a CLS token.
DINOv2 with register tokens (dinov2reg_vit_base_14) has register tokens in the FINAL output
(forward_features) but NOT in the intermediate transformer block outputs that INP-Former uses
for its multi-layer feature extraction.

INP-Former's side calculation:
```python
side = int(math.sqrt(L - 1 - num_register_tokens))
```
where L is the sequence length from intermediate features.

DINOv2 intermediate features have: 780 tokens (784 patches + 1 CLS - 4 register tokens missing)
Expected: 789 tokens (1 CLS + 4 register + 784 patches)
Actual: 780 tokens (missing 4 register tokens in intermediate features)

This causes: sqrt(780 - 1 - 4) = sqrt(775) ≈ 27.8 → 27
But actual tokens = 780, not 27*27=729.

## Solutions

1. **Use DINOv1 or standard ViT encoder** (compatible with INP-Former)
2. **Modify external uad.py** to handle DINOv2 register tokens correctly:
   - Change side calculation to: sqrt(L - num_register_tokens) when CLS not in intermediate features
   - Or pad intermediate features with dummy register tokens
3. **Retrain INP-Former** with DINOv2 register tokens

## Current Workaround

Use the compatible encoder (DINOv1 or standard ViT) with a compatible checkpoint.
The current checkpoint was trained with dinov2reg_vit_base_14 which has this incompatibility.

## Test Status

The SAM+DINOv2 segmenter works correctly (produces meaningful component features).
The INP-Former backbone has a fundamental architectural incompatibility with DINOv2 register tokens.

## Recommended Path Forward

1. **Short term**: Use the compatible INP-Former + DINOv1/ViT setup for experiments
2. **Medium term**: Modify external uad.py to handle DINOv2 register tokens
3. **Long term**: Retrain INP-Former with DINOv2 register token support
"""