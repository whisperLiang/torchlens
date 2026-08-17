# CUDA kernel telemetry

Status: documented unstable -- no deprecation shim owed.

Kernel telemetry is an optional, detachable extension of the gated ATen execution profile. It
measures CUDA kernels and memory copies from one concrete `torch.profiler`/Kineto session and
correlates them to observed `AtenOp` rows. It is not a finer portable model graph, an equivalence
key, a grouping key, or replay evidence.

The adapter is CUDA-only. It brackets each redispatched ATen call with a unique profiler marker,
then uses marker containment to identify CUDA runtime calls and Kineto runtime correlation IDs to
follow asynchronous work onto the device. It does not inspect operator-name or kernel-name
substrings. This supports one ATen call launching many kernels and a single fused launch related to
several ATen rows without pretending either shape is one-to-one.

The public record facade is `KernelLaunch`. Its measured fields are `launch_name`, `device`,
`stream`, `duration`, `runtime_correlation`, and `attribution_status`. The computed
`AtenOp.gpu_kernels` view returns rows related to one primitive call; `Op.gpu_kernels` returns their
deduplicated union for one user Op. All names and status tokens are listed in the glossary's
documented-unstable index.

## Availability and lower bounds

If CUDA or CUPTI is unavailable, telemetry returns a single fact-free disclosure row with
`attribution_status="unavailable"`. That row is not a kernel launch and is never counted as one.
An empty tuple is reserved for a profiler session that ran and observed no related device event.

`mode_paused_interior` remains an observation gap. If a parent Op has any such gap, its observed
kernel set and every count derived from it are lower bounds. TorchLens does not invent a hidden ATen
row, kernel, duration, stream, or correlation ID for the paused region.

## Persistence and shed boundary

Launch rows and their private primitive-sequence relation are declared `FieldPolicy.DROP`. Their
future persistence path is registered through the prerelease registrar and can round-trip only
under the pytest activation switch; ordinary tlspec v7 artifacts omit the telemetry annotation.

No package module imports the adapter, and no non-telemetry behavioral test depends on it. The
central prerelease-registrar inventory and frozen public-surface oracle are the only governance
references: every DROP-gated writer must be enumerated, and the documented computed properties
must remain visible in the public-surface snapshot. Importing the adapter installs those two views
for that process. This one-way dependency is the mechanical shed boundary: removing the telemetry
lane and its two governance references leaves capture, ATen recording, validation, visualization,
FLOPs, predicates, runnable execution, and persistence acceptance unchanged.
