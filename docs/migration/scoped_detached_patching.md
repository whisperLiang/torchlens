# Detached-reference handling: rescue re-run + mechanical belt

TorchLens installs persistent wrappers around torch callables on the first torch capture. A Python
binding created before that installation—such as `from torch import relu`—can still point to the raw
callable after the torch namespace itself is wrapped.

**The historical sys.modules crawler (and its `patch_policy` rollout) is deleted.** TorchLens no
longer rewrites module attributes broadly, reads module sources, or mutates user model instances to
repair stale bindings. Coverage is now:

1. **Rescue re-run.** A completed capture that carries an escape signal — the tensor-provenance
   warning, an `escape_detector` diagnostic, or an output-attribution failure — is re-run ONCE with
   a `TorchFunctionMode` net armed. The net redirects any stale pre-wrap reference to its exact
   wrapper, so recovered ops are logged with full wrapper fidelity. The primary capture is never
   mode-armed (an armed mode flips torch's fused fast paths, e.g. eval `MultiheadAttention`
   3 ops -> 27 ops), so ordinary captures are byte-identical to earlier releases. The rescue also
   covers holder classes the crawler never reached: closure cells, staticmethods, module-level
   partials, plain object attributes, pre-bound tensor methods, torch-free-source modules, and
   C-held references. The most reliable pattern is still to wrap early: import TorchLens or call
   `tl.wrap_torch()` before creating aliases, closures, partials, or object-held torch callables.
   Then those bindings capture the wrappers directly and no rescue is needed.
2. **Mechanical belt.** A small, per-build DERIVED set of wrapped functions is invisible to every
   `TorchFunctionMode` (zero protocol callbacks, measured at wrap time): on current builds
   `torch.from_numpy`, `torch.frombuffer`, and `torch.Tensor.as_subclass`. A stale reference to one
   of these produces no signal a rescue could trigger on, so module-level attribute references to
   them keep targeted patching, with a conditional reversal ledger restored at
   `torchlens.backends.torch.wrappers.unwrap_torch()`.

## Honesty semantics

- A rescued trace is disclosed: `capture_verified=False`,
  `capture_verification_reason == "mode_rescue_rerun"`, and a session-time `trace.rescue_rerun`
  record naming the trigger and the recovered ops. Rescue captures are never byte-attestable (the
  forward ran twice and mode presence can de-fuse fast paths).
- An escape the net cannot recover — a worker-thread stale reference (modes are thread-local; op
  logging is owner-thread-scoped by design) or a stale reference inside a third-party
  `handle_torch_function` composite body (the protocol pops the mode before the body runs) — is
  DISCLOSED, never silent: `capture_verified=False` with reason `"escape_rescue_unrecovered"`
  unless a more specific verdict (dispatch witness, shadow detector, Dynamo boundary) is already
  present, which stays authoritative.
- An armed completeness witness that VERIFIES the capture outranks the heuristic provenance signal
  (for example an `autograd.grad` boundary is a known no-provenance source); no rescue runs.
- Streaming saves, `out_sink` captures, and halt-predicate partials are not re-runnable; they
  report the escape and skip the rescue.

## Deprecated surface

- `tl.wrap_torch(patch_policy=..., patch_modules=...)` — accepted, warns `DeprecationWarning`,
  ignored.
- `patch_detached_references(...)` — no-op shim returning a zeroed `PatchReport`.
- `clear_patch_detached_references_cache()` — no-op shim.
- Trace fields `detached_patch_policy` / `detached_patch_epoch` — removed.
- The `"scoped_dispatch_witness_not_enabled"` verification reason — removed with the policy
  machinery.

## Shadow detector semantics

`escape_detector="shadow"` observes raw callable execution using exact object/code identity. It
reports a `TorchLensCaptureGapWarning` with the callable, registered export sites, source callsite,
storage hint, short stack, and remediation. Reports also appear in `trace.escape_diagnostics`, and
a diagnostic now also TRIGGERS the rescue re-run (the report from the primary run is preserved in
`trace.rescue_rerun["primary_escape_diagnostics"]`).

Every wrapper-to-original edge uses a one-shot immediate-caller token. Tokens are identity-compatible
with transient Tensor-bound builtins by requiring a Tensor receiver and an exact method name from the
wrapped-method inventory. They never exempt the dynamic duration of a wrapper. Thus a raw descriptor
called by a user callback inside a composite wrapper remains reportable. TorchLens-internal
exceptions, if ever required, must match an exact parent/child/callsite row; the audited table has a
hard budget of 16 entries. Its current rows are exhausted by the two branches of the eight
`torch._jit_internal.boolean_dispatch` pooling functions; each row is restricted to the exact
parent wrapper, child callable, `_jit_internal.py` callsite, and `fn` caller.

The one-shot token also cannot represent a Python inventory original that recursively calls itself
through a pre-wrap reference. For example, a synthetic registered original shaped like
`def original(x): return original(x - 1) if x else x` consumes its wrapper token on the first call,
then the recursive call can self-convict. This class is documented rather than exempted because its
callsite is arbitrary original/user code; a general recursion exemption would become an ancestry
blanket and hide real escaped calls.

Shadow is default-off. When it is enabled, the resulting trace remains unverified even with no
reports (`"shadow_diagnostic_mode"`); pair it with `completeness_witness=True` for a positive
verdict.

## Machine-readable qualification

Live torch traces expose these diagnostic fields:

| Field | Meaning |
| --- | --- |
| `escape_detector_mode` | `"off"` or diagnostic `"shadow"` |
| `escape_diagnostics` | Structured raw-call reports accumulated across forward passes |
| `rescue_rerun` | Session-time rescue disclosure (trigger, recovery, primary diagnostics) |
| `capture_verified`, `capture_verification_reason` | Completeness status |
| `capture_owner_thread_id`, `capture_owner_thread_qualified` | Supported proof-domain owner |
| `capture_thread_count_start`, `capture_thread_count_end` | Cheap Python thread-count tripwire samples |
| `capture_thread_activity_detected` | Whether the count changed across a guarded forward |
| `capture_guard_passes` | Per-active-logging pass index, mode, and owner thread |
| `escape_detector_event_count`, `escape_detector_callback_ns` | Detector event and callback-cost counters |
| `escape_detector_backward_coverage` | `"not_armed"` for deferred backward in this rollout |

Public `tl.record(...)` uses the same guarded forward hot path and mirrors these fields onto the
returned `Recording`.

`capture_verified` and `rescue_rerun` are live diagnostic state and do not survive a `.tlspec`
round-trip: a loaded trace reports `None`/unknown, never a falsely preserved `True`.

## Honest boundaries

| Channel | Now |
| --- | --- |
| Closure, dict/list, unrelated class/default, ordinary Python partial | Rescued on signal; shadow reports when Python exposes the call |
| Saved Tensor method descriptor or Tensor-bound builtin | Rescued on signal; shadow reports via descriptor compatibility |
| Protocol-invisible constructors (`from_numpy`, `frombuffer`, `as_subclass`) | Mechanical belt (module-attr patching; membership derived per build) |
| C `functools.partial` around a C builtin | Known profile blind spot; shadow mode stays machine-readably unverified |
| De-moded `handle_torch_function` composite interiors | Beyond any mode; disclosed `escape_rescue_unrecovered` |
| `DataLoader(num_workers=0)` callback executed inside forward | Owner-thread domain; shadow reports visible escapes |
| Worker process preprocessing before model invocation | Outside the armed model-forward domain |
| Model tensor work delegated to another thread/process | Unsupported; owner-thread qualification applies; escapes disclosed, never silent |
| Deferred `trace.log_backward(...)` / `Recording.log_backward(...)` | Explicitly `not_armed` in this rollout |
| `torch.func` / functorch transform internals | Existing transform boundary warning/marker remains authoritative |

The thread tripwire compares `threading.active_count()` at forward entry and exit. It catches a live
count delta cheaply, but a worker that starts and joins entirely inside the forward can evade that
sample. This is why the guarantee remains explicitly owner-thread-qualified.
