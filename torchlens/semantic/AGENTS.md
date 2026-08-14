# semantic/ - Implementation Guide

## __init__.py
- Re-exports the facet registry surface from `facets.py` plus the `patching` and
  `recipes` submodules; `__all__` is the public facet vocabulary.

## facets.py
- Registry API: `register()` (decorator, `class_name=`/`predicate=` matching),
  `reset()`, `using()` (contextvar-scoped extra recipes), `list()`, `info()`,
  `snapshot()`, `mark_current_registry_as_builtins()`.
- View machinery: `FacetView` (a `Mapping` built per record), `Facet`,
  `AttentionHeadView`, `FacetSpec`, `TransformPrimitive`, `FacetRecipe`,
  `FacetMenuItem`, `FacetRegistrySnapshot`, `FacetCapabilityFlags`.
- Absence is typed, never silent: `MissingFacet`, `MissingFacetError`,
  `MissingGradient`, `AbsenceReason`.
- TransformerLens-style names: `enable_transformerlens_aliases()` /
  `transformer_lens_aliases_enabled()` toggle `_TRANSFORMERLENS_ALIAS_TO_NATIVE`.
- Registry state is module-global (`_REGISTRY`, `_REGISTRY_VERSION`); `using()`
  layers recipes through the `_CONTEXT_RECIPES` contextvar.

## patching.py
- Prebuilt counterfactual helpers: `activation_patch_residual_stream()`,
  `activation_patch_attention_output()`, `activation_patch_attention_heads()`,
  `activation_patch_mlp_output()`, `attribution_patch_attention_heads()`.
- Built on `trace()` plus the `facet` selector from `intervention/selectors.py`;
  `_CounterfactualStateGuard` snapshots/restores RNG and stateful tensors so
  clean/corrupted runs stay comparable.

## reconstruction.py
- Fused-SDPA facet reconstruction: `SDPAReconstruction`,
  `sdpa_reconstruction_spec()`, `find_sdpa_op()`.
- Reconstruction is checked, not trusted: `_reconstruct_checked()` recomputes and
  `_allclose_sdpa()` compares against the captured output; failure yields
  `MissingFacet`, never a wrong tensor.

## recipes/ subpackage
- Builtin recipes live in `attention.py`, `embedding.py`, `mlp.py`, `norm.py`,
  `residual.py`; each function is registered with a `@register(...)` decorator at
  import time (e.g. `gpt2_attention`, `gated_mlp`, `layer_norm`,
  `transformer_residuals`).
- `recipes/__init__.py` declares `BUILTIN_FACET_CAPABILITY_INVENTORY`, calls
  `mark_current_registry_as_builtins()`, then `_load_entrypoint_recipes()` loads
  `torchlens.recipes` entry points fail-safely (only callables flagged
  `_torchlens_recipe_autoload` are invoked; broken plugins warn, never raise).
- `_helpers.py` holds spec builders shared by recipes: `child_output_spec()`,
  `first_input_spec()`, `module_input_op_spec()`, `module_output_spec()`,
  `parameter_spec()`, `reshape_heads()`, `fused_sdpa_facet()`, `config_value()`.

## Local Invariants / Gotchas
- Dual home: top-level `torchlens/facets.py` is a self-replacing stub that swaps
  itself in `sys.modules` for `torchlens.semantic.facets`, so the two module
  objects are identity-equal; the `tl.facets` attribute resolves lazily via
  `_LAZY_ATTRS` in `torchlens/__init__.py`. Edit `semantic/facets.py` only.
- Builtin recipes register when `torchlens.semantic` is first imported (run
  time), not at `import torchlens`; tests must not assume collection-time
  registration.
- Records expose facets through the `facets` property on `Op`, `ModuleCall`,
  and `Module` (each caches one `FacetView`).
- `list` shadows the builtin in `facets.py`; internal code must use
  `builtins.list`.
