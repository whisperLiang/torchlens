# report/ - Implementation Guide

Reporting helpers over finished captures and observer metadata (`tl.report`).

## Files

- `_explain.py` — `explain(log, ...)`: the prose explainer for a Trace / partial
  trace (used as `tl.report.explain(log)` in the root docs' Common Patterns).
- `_profile.py` — `TraceProfile` + `build_profile(...)`: tabular per-layer/module
  profile (durations, FLOPs, params, honesty rows, call tree). pandas is an optional
  dependency resolved lazily via `_require_pandas()` — never import it at module top.
- `__init__.py` — public surface (`explain`, `TraceProfile`, `build_profile`,
  `log_value`); `log_value(name, value)` records observer values through `_state`.

## Gotchas

- Honesty rows must reflect capture verification state faithfully — a rescued or
  ceilinged capture (`capture_verified=False`) must stay visible in profile/explain
  output; never present an unverified capture as clean.
- `forward_peak_memory` is a runtime measurement that legitimately reads `0` on the
  default CPU path — never present it as a portable fact (root `CLAUDE.md` rule).
