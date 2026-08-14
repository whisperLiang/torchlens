# bridge/ - Implementation Guide

Optional adapters between TorchLens traces and external tools. The package
`__init__.py` lists every adapter in `_BRIDGE_MODULES` and imports them lazily
via module-level `__getattr__`; `import torchlens.bridge` itself pulls in no
optional dependency.

## _utils.py (shared helpers)
- `source_model()` returns the live model from `trace._source_model_ref` and
  raises `ValueError` when the model was garbage-collected.
- `resolve_one_site()`, `out_at()`, `first_input_tensor()`, `tensor_layers()`
  map site-like values (labels/selectors/layer objects) to saved tensor outs.

## Adapter files and main entry points
- `captum.py`: `attribute()`, `layer()` (extra: `torchlens[captum]`).
- `shap.py`: `explain()` (default `shap.DeepExplainer`; extra `torchlens[shap]`).
- `sae_lens.py`: `encode()`, `decode()` (extra: `torchlens[sae]`).
- `lit.py`: `TorchLensLitModel`, `model()` (extra: `torchlens[lit]`).
- `hf.py`: `trace_text()`, `trace_image()`, `trace_multimodal()` plus input
  detection helpers (`_is_hf_text_input`, `_is_hf_image_input`,
  `_is_hf_multimodal_input`, ...). This is the autoroute bridge: consumed by
  `autoroute/_builtin_input.py` and imported eagerly by `user_funcs.py`, so it
  must import cleanly without `transformers` installed (gating stays inside
  the functions).
- `huggingface.py`: `push_to_hub()` for artifacts (extra: `torchlens[hf]`).
- `profiler.py`: `execution_trace()`, `join()` correlate a Kineto/Chrome trace
  with captured layers; stdlib-only, no import gate.
- `gradcam.py`: `cam()`, `layer()` (extra: `torchlens[gradcam]`).
- `brain_score.py`: `per_layer()` takes a CALLABLE offline benchmark (no
  import gate; raises `TypeError` on non-callables).
- `rsatoolbox.py`: `dataset()` (extra: `torchlens[neuro]`).
- `nnsight.py`: `from_trace()` normalizes a cached nnsight-style trace into a
  stable payload schema; offline, no import gate.
- `inseq.py`: `attribute()` (extra: `torchlens[inseq]`).
- `depyf.py`: `dump()` (extra: `torchlens[depyf]`).
- `dialz.py`: `analyze()` (extra: `torchlens[dialz]`).
- `repeng.py`: `control_vector()` (extra: `torchlens[repeng]`).
- `steering_vectors.py`: `vector()` (extra: `torchlens[steering]`).

## Optional-dependency gating pattern
- Never import an optional dependency at module top level. The pattern is a
  function-local `try: import x / except ImportError: raise ImportError(...)`
  naming the exact extra, e.g.
  "Captum bridge requires the `captum` extra: install torchlens[captum].".
- Tests gate on the dependency with `pytest.importorskip()`; offline adapters
  (`brain_score`, `nnsight`, `profiler`) run without extras.

## Local Invariants / Gotchas
- Adding an adapter requires updating BOTH `_BRIDGE_MODULES` and `__all__` in
  `__init__.py`; a name missing from `_BRIDGE_MODULES` raises
  `AttributeError` on access.
- Bridges that execute the model (`captum`, `shap`, `gradcam`, `repeng`,
  `steering_vectors`, ...) need the source model alive; `tl.release_model()`
  or a dropped reference makes `source_model()` raise.
- Site arguments must carry saved tensor outs; `out_at()` raises `ValueError`
  otherwise. Keep error messages actionable (name the site and the extra).
