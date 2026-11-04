# Effective split state bindings

`SplitRuntime.state_bindings()` exposes the effective executed Torch state as
read-only mappings from stable state ID to owner (`prefix` or `suffix`) to live
tensor. It includes parameters, running buffers and anonymous literal sources.
The IDs and owners correspond to `parameter_ownership()`.

Calling it settles lazy segment state. It does not step optimizers, clear
gradients, copy values into the source model, or promise that the source model
contains updates made by owned replicas on another device. Consumers must
deduplicate tied aliases and must not combine diverged owner values silently.

`parameter_bindings()` keeps its existing parameter-only contract for external
optimizers. Both methods currently require the Torch split adapter; other
backends raise the existing typed state-ownership refusal.

For independent prefix/suffix executors, use
`local = runtime.bind_boundary_state(received_boundary)` before suffix training.
It validates the received contract and rebinds registered source buffers to the
suffix's effective state. It leaves computed activations and buffer observations
alone. This explicit operation prevents stale client copies of server-owned
BatchNorm statistics from resetting server state between batches.

Torch single-batch suffix training also accepts `boundary_gradients_only=True`
for calibration VJPs without parameter-gradient accumulation, and
`phase_timer(name)` context-manager hooks for `suffix_forward`, `suffix_loss`
and `suffix_backward`. A boundary-only VJP cannot step an optimizer. Other
backends and suffix microbatches refuse these optional execution controls.
