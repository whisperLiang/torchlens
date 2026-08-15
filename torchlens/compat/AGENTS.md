# compat/ - Implementation Guide

`tl.compat.report(model, x)` is the FIRST diagnostic the root docs tell a debugging
agent to run — keep this package cheap to import and side-effect free.

## Files

- `_report.py` — `CompatRow` / `CompatReport` dataclasses and `report(model, input)`:
  the runtime support report (wrapper coverage, capability flags, distributed findings
  such as the `dtensor` / `device_mesh` / `tensor_parallel` / `pipeline_parallel` rows).
  `CompatReport.to_markdown()` is the documented rendering.
- `torchextractor.py` — `Extractor`, the torchextractor-style adapter (callable and
  `forward` both return a `dict[str, torch.Tensor]` of layer outputs).
- `lovely.py` / `torchshow.py` — thin adapters that coerce TorchLens records to tensors
  for the lovely-tensors and torchshow display libraries. They are lazy compat modules
  (`_COMPAT_MODULES` in `__init__.py`); their third-party deps are optional and must be
  imported only inside the adapter calls.

## Gotchas

- Capability truth comes from feature-detected `HAS_*` flags (see
  `utils/_torch_compat.py`); never parse `torch.__version__` here.
- The distributed rows share detection with capture entry via `torchlens/_distributed.py`
  — do not fork the logic; both surfaces must agree.
- Lazy `HAS_*` probe caches are process-global; tests that stub `sys.modules` must
  restore the probe state or later compat snapshots flap.
