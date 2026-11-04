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
