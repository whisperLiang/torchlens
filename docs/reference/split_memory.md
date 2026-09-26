# Split execution memory and microbatch training

These split APIs are documented-unstable. A prepared runtime keeps its canonical,
batch-polymorphic graph and can replay or train different logical batch sizes without
retracing. Torch suffix microbatching is an execution option on that same runtime.

## Training one logical batch

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.split import SplitFeatures, SplitRequest, after

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
x = torch.randn(7, 4)
targets = torch.randn(7, 3)
runtime = tl.split.prepare(
    model, x,
    SplitRequest(point=after("1"), features=SplitFeatures(training=True)),
)
prefix_optimizer = torch.optim.SGD(model[0].parameters(), lr=0.01)
suffix_optimizer = torch.optim.SGD(model[2].parameters(), lr=0.01)

boundary = runtime.run_training_prefix(x)
loss, boundary_grads = runtime.train_suffix(
    boundary, targets, optimizer=suffix_optimizer, microbatch_size=3,
)
runtime.backward_prefix(boundary, boundary_grads, optimizer=prefix_optimizer)
```

This executes one prefix for seven samples, suffix chunks of `3 + 3 + 1`, and one
prefix backward using the assembled logical-batch gradients. A supplied suffix optimizer
receives one `zero_grad(set_to_none=True)` and one `step()`. The caller owns the optimizer;
preparation and prefix execution do not allocate optimizer state.

For suffix-only adaptation, use `boundary = runtime.run_prefix(x)` and omit
`backward_prefix`. `run_prefix` executes Torch operations under `torch.no_grad()` and
returns detached boundary tensors. They are ordinary tensors that suffix autograd can
save. `run_training_prefix` preserves the caller's grad mode and graph connection for
prefix backward, so call it with grad mode enabled when training the prefix.

For repeated inference where the caller keeps model and segment state stable between
calls, `runtime.run_prefix(x, check_state=False)` followed by
`runtime.run_suffix(boundary, check_state=False)` skips the full state fingerprint on
both sides. Graph, shape, dtype, and boundary identity checks still run. The default
`check_state=True` detects stale reusable boundaries after a state update; it rejects
a boundary created with `check_state=False`. Use the default for cached boundaries
and training.

Both `train_suffix` and `train_suffix_result` accept keyword-only options:

| Option | Default | Meaning |
| --- | --- | --- |
| `microbatch_size` | `None` | Positive integer maximum chunk size; `None` keeps existing full-batch execution. |
| `microbatch_reduction` | `"mean"` | `"mean"` weights each chunk loss by its sample fraction; `"sum"` accumulates unscaled custom sum losses. |
| `target_slicer` | `None` | Optional `(targets, start, end, logical_batch) -> chunk_targets` callback. |

Microbatch execution returns a detached logical loss. `train_suffix_result` provides the
same loss and gradients in `TrainingStepResult`, together with optimizer-step status.
Other backends reject a non-`None` microbatch size with `SplitUnsupportedError`.

The default full-batch `train_suffix` path validates the caller-owned boundary once,
then runs the root-swapped suffix directly. It does not rehash or revalidate that
internal boundary for the same step. Public `run_suffix` keeps strict validation for
boundaries that may have been cached or reused after a state update.

Strict prefix/suffix calls compute a value-sensitive state digest on each call, so
updates through `.data` or storage aliases still invalidate reusable boundaries.
This requires reading the effective state, including a device-to-host copy for CUDA
state. For one-shot inference with stable state, use `replay()` or explicitly opt
into `check_state=False` on both public calls to skip that cost.

## Loss and slicing semantics

A custom `loss_fn(output, chunk_targets)` must return a real scalar tensor. For mean
reduction, chunk `i` with `b_i` samples contributes `loss_i * b_i / B`. Thus an uneven
last chunk has its correct weight. Sum reduction requires an explicit sum-reduced
`loss_fn`, and each chunk contributes its unscaled loss.

Full-batch equivalence requires independent samples and an additive objective. Mean
losses must have a denominator proportional to sample count. Training BatchNorm,
cross-sample objectives, and stochastic operations such as dropout can change behavior
when chunked. Losses normalized by a variable number of unmasked tokens, class weights,
or `ignore_index` counts may need a custom normalization and sum reduction. The runtime
cannot infer these semantics from an arbitrary loss callable.

Boundary slicing uses the existing `ShapeProgram`, including nonleading batch axes.
A tensor with one plain batch symbol `B` is sliced on that axis. A value without `B`
is reused logically in every chunk. Compound or multiple batch axes and unresolved
boundary shapes refuse with structured `SplitUnsupportedError`; extents alone never
establish boundary batch semantics. Existing shape eligibility is checked for every
chunk size before optimizer gradients are cleared.

Targets follow a separate small convention:

- Tensor leaves whose leading extent equals logical `B` are sliced.
- Mappings and tuples are traversed recursively; namedtuple types are preserved.
- Lists of length `B` represent samples and are sliced as lists; other lists are traversed.
- Scalars and other nonbatched values are reused.

Use `target_slicer` when these conventions are ambiguous, for example a list of `B`
feature tensors that should be sliced internally. Mappings are rebuilt as ordinary
dictionaries. Targets must already be on the device required by the loss.

## Lifetimes and placement

Each suffix chunk creates independent autograd roots. Its backward finishes before the
next graph starts; outputs, losses with graphs, and roots are not retained across chunks.
Batched boundary gradients are copied into one preallocated logical-size tensor in
original order. Nonbatched differentiable boundary gradients are summed across chunks.
Chunk inputs are isolated from the caller's boundary storage before replay operations.

Replay continues to use the existing last-consumer release schedule. Residual values
remain available through their actual final consumer, and multioutput calls retain
needed outputs. Public `run_suffix(boundary)` leaves the boundary object available for
reuse. Internal `replay(inputs)` releases its source boundary reference after transport,
before suffix execution.

For CPU prefix / CUDA suffix, prepare with
`PlacementPlan.across("cpu", "cuda")`. Microbatch training transports the detached
logical boundary once, then slices it on the suffix device. Returned gradients are on
the suffix device; `backward_prefix` moves them to the connected prefix tensors as needed.
Caller-owned source boundaries remain alive while the caller retains them. Graph-connected
prefix activations and the final logical boundary gradient still consume full-batch memory.

## State sharing

Each runtime's segment bundle owns a replica pool; there is no global tensor cache.
Replicas are keyed by backend, effective source identity, and normalized destination
device. Tied uses of one source retain one identity within each segment.

Distinct bindings share a replica only when it is frozen and all captured uses are
certified read-only operations that cannot return aliases of their parameter inputs.
The current conservative table covers linear and convolution operations. An unknown
use, a view or `detach`, a trainable storage alias, or a buffer alias excludes that storage.
For example, a frozen `Embedding(max_norm=...)` still modifies its weights and stays
independent. Mutable buffers, unproven literals, and separately owned trainable replicas
are not pooled. Different destination devices always have distinct replicas.

A live inference prefix and its training prefix use the same binding when their state
policy is identical. A captured-state inference prefix remains separate from its live
training prefix. Recutting and placement changes preserve effective, possibly updated
state; pooling never substitutes the original source for an inherited updated value.
Existing same-device live references keep their established sharing semantics.
Live registered buffers follow replacements made by `model.to(...)` without growing
the state table. Independently owned buffer replicas retain their effective values
when the source model moves, including across recuts. Compact runtimes use weak owner
references, so following live buffers does not retain the source module or its Trace.

Microbatch size does not enter graph identity, split identity, the shape program, or the
semantic boundary cache ABI. A saved logical boundary can be loaded and trained with a
different chunk size. Memory savings depend on the model: activation-heavy suffixes
benefit most, while parameter gradients, optimizer state, retained captures, and the
logical boundary impose a remaining memory floor.

## Optional CUDA measurement

The opt-in regression uses an activation-heavy convolutional model, warms both execution
paths, clears gradients and results, synchronizes CUDA, and measures allocated peaks
with `reset_peak_memory_stats()` / `max_memory_allocated()`. It repeats both comparisons;
allocator reservations are not the measured quantity.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 \
TORCHLENS_CUDA_MEMORY_TESTS=1 .venv/bin/pytest \
  tests/split/test_split_memory_cuda.py -q -s
```

Run the command in a fresh process with exclusive GPU use for comparable measurements.
The test checks that microbatch suffix training and detached prefix execution each use
less peak allocated memory than their corresponding full/connected modes, without a
fixed percentage threshold.
