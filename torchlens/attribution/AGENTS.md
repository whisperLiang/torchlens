# attribution/ - Implementation Guide

Input-attribution methods over live models (lazy public submodule,
`tl.attribution`). Gradient-based attribution runs the USER's model directly
(with a temporary `eval()` context and fresh input leaves), not a replay of a
captured Trace.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public re-exports only; the 10-name `__all__` below |
| `_core.py` | `AttributionError`, `AttributionResult`, input normalization / leaf minting / baseline validation machinery, and the input-level methods (`saliency`, `input_x_grad`, `integrated_gradients`, `smoothgrad`) |
| `_layer.py` | Layer-level methods: `layer_attribution`, `layer_integrated_gradients`, `layer_conductance`, `grad_cam` |

## Public surface (`__all__`, 10 names)

`AttributionError`, `AttributionResult`, `saliency`, `input_x_grad`,
`integrated_gradients`, `smoothgrad`, `layer_attribution`,
`layer_integrated_gradients`, `layer_conductance`, `grad_cam`.

## Gotchas

- `AttributionError` subclasses `ValueError`; argument/baseline problems
  refuse through it, not bare exceptions.
- Inputs are normalized through `_PreparedInputs` and replaced by fresh
  requires-grad leaves (`_make_input_leaves`); nested input trees keep
  identity-interning so repeated tensor objects stay aliased.
- Baselines are validated against the exact input tree shape
  (`_validate_baseline_tree`); repeated-reference baselines have their own
  validation path.
- The model's `training` flag is saved/restored by `_temporarily_eval` —
  do not add code paths that return without restoring it.

## Tests

`tests/test_attribution.py`, `tests/test_attribution_layer.py`,
`tests/test_backward_attribution_unchanged.py`.
