# errors/ - Implementation Guide

Public TorchLens exception classes. Public code branches on exception TYPE and
structured fields (`exc.fields[...]`), never on message text; the exhaustive runnable
error vocabulary and release threshold live in
`docs/reference/runnable_tlspec_contract.md`, and the error/refusal contract is
`docs/reference/error_refusal_contract.md`.

## Files

- `_base.py` — the base hierarchy: `TorchLensError`, `CaptureError`,
  `CompatibilityError`, `ConfigurationError`, `ValidationError`, `InterventionError`,
  `TorchLensWarning`, `Severity`, and friends. Everything public derives from
  `TorchLensError`.
- `runnable.py` — the runnable/tlspec family: `RunnableTLSPECError` and its typed
  subclasses (`RunnablePreflightError`, `SparseCorePayloadError`,
  `RunCapabilityUnavailableError`, `ReattachError`, `StateBindingError`,
  `RunPreconditionError`, `RuntimeSignatureDriftError`, ...). Each subclass mixes in
  the correct base-category class AND the matching builtin (`ValueError` /
  `RuntimeError` / `AssertionError`) — preserve both bases when adding one.
- `__init__.py` — the public re-export surface (`__all__`) plus
  `_LAZY_EXCEPTION_PATHS` legacy path aliases resolved through `importlib`.

## Gotchas

- Shared internal exception helpers live in `torchlens/_errors.py` (built ON
  `errors._base`), and the intervention error catalog in
  `intervention/errors.py` — three homes by design; see the dual-homes note in
  `.project-context/architecture.md`.
- New stable refusal codes ride `exc.fields["code"]` (e.g.
  `tl.errors.CaptureOutcomeError`); update the contract doc and its lockstep test in
  the same change.
