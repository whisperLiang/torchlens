# export/ - Implementation Guide

One-module export bridge (`tl.export`, lazy): renders/serializes a FINISHED
trace (or bundle) into third-party formats. Everything lives in
`__init__.py` (17-name `__all__`).

## Surface (grouped)

- Visual: `svg` (editable Graphviz SVG), `html` (self-contained page).
- Profiling: `chrome_trace`, `chrome_trace_diff` (bundle), `speedscope`,
  `flamegraph`, `memory_timeline`.
- Tabular/array: `csv`, `parquet`, `json`, `xarray`.
- Experiment trackers: `tensorboard(log, writer)`, `wandb(log, run)`,
  `mlflow(log, client)`, `aim(log, run)` — the tracker OBJECT is passed in
  (guarded by `_require_tracker_object`, duck-typed on the required method);
  this package never imports tracker SDKs itself.
- Model viewers: `model_explorer`, `netron`.

## Gotchas

- File-writing exporters take an explicit `path` and return the written
  `Path`; nothing writes to implicit locations.
- Optional heavy dependencies (pandas/pyarrow/xarray/graphviz consumers)
  resolve lazily at call time; add new exporters with the same deferred
  pattern, never module-top imports.
- These renderers are NOT behind the capture-outcome N-gates (those cover
  `tl.save`/replay/validation entries); they read whatever the log exposes.
  Report-side honesty (unverified/halted disclosure) lives in `report/`.

## Tests

`tests/test_exports.py`, `tests/test_export_behaviors.py`,
`tests/test_export_html_minimal.py`, `tests/test_io_export.py`.
