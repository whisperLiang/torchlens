# experimental/ - Implementation Guide

Incubating APIs with UNSTABLE naming and behavior (`tl.experimental`).
Nothing here is covered by the public deprecation policy; expect renames.

## Files

| File / dir | Purpose |
|------------|---------|
| `__init__.py` | The experimental helpers + lazy submodule loader (10-name `__all__`) |
| `dagua/` | Experimental Dagua rendering bridge (`_bridge.py`); import it before selecting `vis_renderer="dagua"` |
| `node_styles.py` | Experimental node styling helpers |

## Surface (`__all__`)

`attribute_walk(model, address)`, `stop_after(site)` (context manager),
`Session` / `session(model)`, `AutoCaptureSession` /
`auto_capture(model, every=100)` (context manager), `freeze_module(layer)`
(context manager), plus the `dagua` and `node_styles` submodules (lazy via
module `__getattr__`).

## Gotchas

- Graduation path: a helper that stabilizes moves OUT of this package with
  a deprecation shim left behind; never let core code import from
  `experimental/`.
- The Dagua renderer is opt-in and experimental; the production renderer
  remains Graphviz (`visualization/`). See `docs`/root notes on the planned
  dagua integration before extending `_bridge.py`.

## Tests

`pytest -k "experimental or dagua"` over `tests/`.
