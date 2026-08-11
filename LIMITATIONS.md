# TorchLens Limitations

TorchLens observes eager PyTorch execution by wrapping PyTorch call sites and recording what
actually runs. Its substrate identity is therefore "PyTorch eager execution plus TorchLens
metadata", not a separate static IR, compiler graph, or symbolic model format.

## Single-Threaded By Design

TorchLens is single-threaded by design. Capture depends on process-global logging state and
ordered operation counters, so running captures concurrently in multiple Python threads or child
processes can corrupt ordering assumptions. Use TorchLens from the main process.

## Dynamic Control Flow

Eager dynamic control flow is a feature, not a limitation. TorchLens records the branch, loop, and
module behavior that occurred for the concrete input you supplied. It does not claim to enumerate
branches that did not execute.

## torch.compile Regions and Tracing Tensors

Tensors inside a `torch.compile` (Dynamo) region are data-free `FakeTensor`s. Every TorchLens step
that reads a value -- `safe_copy`, `torch.equal`, `.item()`, `data_ptr()`, memory accounting -- is
meaningless or fatal on them, so TorchLens cannot record real activations there.

Compiled child `nn.Module`s are handled automatically: they are swapped for their eager source
module for the duration of capture (with a one-time note), so their interiors *are* logged. A
compiled **callable** held as a plain attribute or called as a free function cannot be swapped out.
Reaching one during capture used to die with a raw
`torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'FakeTensor' object has no attribute
'fake_mode'`. It now degrades gracefully: operations inside the compiled region are not logged,
a one-per-forward `UserWarning` names the gap and the remedy, and the returned `Trace` contains
only what ran outside the region, marked `capture_verified=False` with
`capture_verification_reason="dynamo_region_not_logged"`. Call the eager function during capture
if you need its interior.

Passing a `FakeTensor` or `FunctionalTensor` as a model input, or tracing a model whose parameters
were built under a fake mode, is refused at capture entry with `UnsupportedTensorVariantError`
alongside the other data-free variants. Previously this crashed mid-forward with a bare
`AssertionError: Please convert all Tensors to FakeTensors first` from torch's own fake machinery,
after TorchLens had already tripped torch's "almost definitely a bug in your code" warning by
reading a FakeTensor's `data_ptr()`.

## Retained Activation Footprint

`tl.trace(model, x)` retains every operation's output by default. That is the right default for
the models TorchLens was designed around and a footgun at frontier shapes, where it means an OOM
kill (or an allocator error from deep inside torch) rather than an explanation.

Capture therefore enforces a per-device ceiling on retained payload bytes, `save_budget`, which
defaults to half of each device's *available* memory measured at that device's first save. Crossing
it stops capture with `torchlens.errors.SaveBudgetExceededError`, naming the bytes committed so far,
the budget and where it came from, the operation that tripped it, and the remedies.

The reported footprint is an explicitly-labelled **lower bound**: the forward pass was still running
when the budget tripped, so a completed default capture would have retained more. TorchLens says
that rather than extrapolating a total it cannot know.

Only payloads actually held in RAM are charged, so `storage=tl.to_disk(...)` and
`layers_to_save="none"` are never budgeted -- those captures were never going to exhaust memory.
Devices whose headroom cannot be measured are left unbudgeted and reported as such rather than
silently assumed infinite.

Tune it with `capture=tl.options.CaptureOptions(save_budget=...)`: a float in `(0, 1]` for another
fraction of available memory, an int for an absolute per-device byte cap, or `None` to disable the
guard. A malformed value raises rather than silently unguarding the capture.

## Distributed and Sharded Execution

TorchLens captures a **rank-local eager forward pass**. Sharded distributed execution is out of
scope for this release, and TorchLens now says so instead of returning a wrong trace.

`DTensor` and `ShardedTensor` are tensor subclasses whose real work happens under
`__torch_dispatch__`, *below* the `__torch_function__` layer TorchLens wraps. A capture of such a
model does not crash: it silently records rank-local view/reshape shims in place of the real ops
and reports zero parameters. Pipeline-parallel stages are a related case — a stage holds only its
own slice of the model and is driven by a schedule that runs microbatches outside the traced
forward, so a capture there is a fragment, not the model.

Because all-pass-then-silently-wrong is worse than a clear failure, both surfaces now report it:

- `tl.compat.report(model, x)` has `dtensor`, `device_mesh`, `tensor_parallel`, and
  `pipeline_parallel` rows.
- `tl.trace(...)` refuses at capture entry with
  `torchlens.errors.DistributedCaptureUnsupportedError`, naming the exact parameter, buffer, or
  input sites that carry the sharded state. The structured findings are on
  `exc.fields["findings"]`, so code branches on `finding.kind` rather than message text.

Refusal is deliberately narrower than reporting. Only the conditions that provably corrupt a
capture refuse: sharded tensor state (`dtensor`) and pipeline stages (`pipeline_parallel`). A bare
`DeviceMesh` attached to a model whose parameters are ordinary dense tensors, or a tensor-parallel
style wrapper that left dense parameters behind, is reported as a warning row and captured
normally, because that capture is correct.

**What to do instead:** trace the undistributed module. Build and trace the model before wrapping
it with `parallelize_module` / `fully_shard` / pipeline stages, or materialize the logical tensors
(`DTensor.full_tensor()`) into a plain module first. Cross-rank merging (`tl.merge_ranks()`) is not
available in this release.

Detection is capability-probed, never version-parsed, and every degradation point is visible as a
named flag in `torchlens.utils.doctor()` / `tl.compat.report()`: `HAS_DTENSOR`, `HAS_DEVICE_MESH`,
and `HAS_PIPELINING`. When one is absent, detection falls back from an exact `isinstance` check to
structural namespace matching, and the affected report row says so.

## Bundle Diff Rendering

`Bundle.show_diff()` is static for v1. Interactive bundle comparison belongs in
a future local inspection appliance.

## Compatibility Truth Table

`tl.compat.report(model, x)` reports the following rows at runtime. Rows marked
`known_broken` or `scope` here should be included when filing issues so maintainers can
separate unsupported contexts from new TorchLens bugs.

| Row | Status | Notes |
| --- | --- | --- |
| `hf_transformers` | `pass` when detected alone | Eager Hugging Face modules are supported when not compiled, sharded, offloaded, or quantized with custom kernels. |
| `accelerate_device_map_auto` | `known_broken` when detected | `device_map="auto"` can materialize parameters across devices during forward. |
| `accelerate_cpu_disk_offload` | `known_broken` when detected | CPU/disk offload hooks mutate device placement lazily. |
| `bitsandbytes_8bit_4bit` | `known_broken` when detected | bitsandbytes parameter wrappers and custom kernels are outside the dense eager tensor contract. |
| `tied_parameters` | `pass` | Shared parameter objects are detected and reported. |
| `multi_gpu_rng` | `pass` | RNG snapshots now use all visible CUDA devices; include this row for multi-GPU bugs. |
| `data_parallel` | `known_broken` when detected | `nn.DataParallel` uses threaded replicas and conflicts with process-global capture state. |
| `distributed_data_parallel` | `pass` when detected | DDP unwraps to the rank-local `.module`. |
| `fsdp` | `scope` when detected | FSDP sharded materialization is not launch-scope support. |
| `dtensor` | `scope` when detected | DTensor/ShardedTensor state; capture refuses with `DistributedCaptureUnsupportedError`. |
| `device_mesh` | `scope` when detected | Informational: identifies the distributed topology. Does not refuse capture on its own. |
| `tensor_parallel` | `scope` when detected | Rank-local shapes and parameter counts are fractions of the logical model; collectives are invisible. |
| `pipeline_parallel` | `scope` when detected | A stage is a fragment driven by an external schedule; capture refuses. |
| `deepspeed` | `scope` when detected | DeepSpeed/ZeRO/offload execution is not launch-scope support. |
| `torch_compile` | `scope` when detected | Log the eager model; use `torchlens.bridge.depyf` for compiled-code context. |
| `fx_graph_module` | `scope` when detected | TorchLens targets eager execution, not FX IR parity. |
| `lightning_training_step` | `known_broken` in active training mode | Use the Lightning callback or log a plain forward outside the trainer loop. |
| `vmap_functorch` | `known_broken` when detected | TorchLens skips logging inside active functorch transforms and returns incomplete logs. |
| `quantized_tensor` | `known_broken` when detected | Quantized comparison no longer crashes, but full quantized activation metadata remains best-effort. |
| `device_context_factory` | `pass` | Factory functions honor active `torch.device(...)` contexts during logging. |
| `single_thread_design` | `known_broken` off main process/thread | Capture state is process-global and ordered for a single main-thread forward. |
| Static graph export parity | `known_broken` | TorchLens records eager execution, not static export IR. |
| Optional bridge without its extra | `known_broken` | Install the matching extra before using bridge adapters. |

See `ROADMAP.md` for planned follow-up work.
