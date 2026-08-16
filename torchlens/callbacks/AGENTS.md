# callbacks/ - Implementation Guide

Trainer-framework callback integrations. The namespace is LAZY by design:
`torchlens.callbacks.lightning` imports only on attribute access
(`_CALLBACK_MODULES` gate in `__init__.py`), so `import torchlens` never
pays for (or fails on) an absent trainer framework.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | PEP-562 lazy loader over the callback module set (`{"lightning"}`) |
| `lightning.py` | `LayerProfilerCallback` (the only `__all__` name) — built at ACCESS time by `_build_layer_profiler_callback()` on top of Lightning's callback base |

## Gotchas

- `LayerProfilerCallback` does not exist as a static class: `lightning.py`'s
  own module `__getattr__` resolves it lazily so the Lightning import (and
  its optional-extra refusal) happens on first use, never at module import.
  This is the same deferred-extras pattern as the appliance packages — do
  not "fix" it to import Lightning at the top.
- `_LayerProfilerCallbackMixin` holds the framework-agnostic logic; the
  Lightning base is grafted on in `_build_layer_profiler_callback()`.

## Tests

`tests/test_callbacks_lightning.py` (uses `pytest.importorskip` for the
optional dependency).
