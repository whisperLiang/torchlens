# bundle/ - Implementation Guide

Single-module package (`__init__.py`) exporting `Bundle` and `AmbiguousLabelError`
(`__all__` at the bottom of the file).

- `Bundle` is a named collection of aligned Traces for cross-run comparison,
  constructed by `tl.bundle(...)` (top-level convenience in `torchlens/__init__.py`)
  or directly as `tl.Bundle(...)`.
- `AmbiguousLabelError` fires on cross-member label lookups that match more than one
  aligned site.
- The bundle namespace is dual-homed by design: rendering for bundles lives in
  `visualization/bundle_diff.py` (`tl.show_bundle_graph` routes through
  `user_funcs.py` / `_user_public_impls.py`), not here. Keep graph/render logic out
  of this package.
- Public-name changes here must update the glossary and root docs in the same change
  (see the lockstep rule in the root `CLAUDE.md`).
