# Split Runtime on TorchLens 3.x

`3.0.3+split.1` combines the upstream 3.x public core with the fork's native
Split Runtime. Existing Split callers continue to import from `torchlens.split`.
The [executable example](../../examples/split_upstream3.py) prepares one graph,
selects an explicit partition, replays several batches and changes the partition.

## Capture and execution

Capture settings belong in `tl.options.CaptureOptions`, for example:

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(4, 3).eval()
inputs = torch.randn(2, 4)
capture = tl.trace(
    model,
    inputs,
    capture=tl.options.CaptureOptions(replayable=True, save_args=True),
)
```

For Split execution, call `prepare_graph(model, inputs)` instead. It requests
the capture facts the native adapter needs and projects the upstream records
once. `op.call` owns the call identity and argument templates, `op.graph` owns
graph facts, `op.params` owns parameter metadata, and `op.cost` owns cost facts.
The public records keep this upstream schema. Code importing former internal
modules should use public Split APIs; the former public implementation packages
are not installed alongside the 3.x core.

`graph.compute_dag()` returns native calls, data edges, storage effect edges
and groups that must share an executor. An external scheduler selects call IDs;
`dag.to_partition(call_ids)` expands them into `GraphPartition.prefix_node_ids`.
`graph.analyze(partition)` proves legality and `graph.materialize(analysis)`
builds prefix and suffix execution. A multi-output native call stays atomic.
Repeated output leaves retain their container paths and share transported values.

Canonical IDs are stable through analysis, materialization and repartition of
one captured graph. They are not display labels, and equal-looking IDs from
independent captures do not establish that their values are interchangeable.

Replay executes the captured native calls. Live parameter and buffer bindings
refer to execution state, rather than saved diagnostic activations. Batch changes
use the independent probe and shape program; a shape change the runtime cannot
establish is refused. Training preserves boundary gradients, optimizer bindings
and current state through recuts. See [state bindings](../reference/split_state_bindings.md)
and [training and memory](../reference/split_memory.md).

## Native backend boundaries

| Backend | Capture and Split binding |
| --- | --- |
| PyTorch | Upstream grouped call records, generated eager replay, storage effects, RNG and autocast context; live parameters and buffers |
| JAX | Upstream jaxpr equations, native parameter PyTrees and gradients; dynamic equation operands use the shape program |
| tinygrad | Upstream structural UOp capture with one realization, exact operand paths and live model tensor bindings; fixed-shape native replay |
| MLX | Upstream wrapped calls, native module and array state, witnessed state writes and per-call random state |
| TensorFlow | Upstream eager kernels, native Tensor/Variable resource bindings and GradientTape execution |
| Paddle | Upstream native calls with exact parameter and registered-buffer operand paths; native device context is scoped around capture and validation |

These bindings do not make every operation supported. `analyze()` and the
adapter's structured capability report remain the authority for a concrete
partition. A validation failure still identifies missing capture or replay
evidence; it is not an instruction to bypass a tripwire.

Native capture refusals retain a structured `BackendUnsupportedError.code`:

| Code | Cause and remedy |
| --- | --- |
| `module_identity_ambiguous` | tinygrad set members cannot receive distinct stable module addresses; use an ordered container or distinct names |
| `cuda_driver_unavailable` | Paddle CUDA context inspection cannot load the NVIDIA driver; restore the driver installation |
| `cuda_context_inspection_failed` | A CUDA driver context query or switch failed; inspect the reported operation and driver status |
| `cuda_context_unavailable` | Paddle GPU storage has no owning CUDA context; recreate the native tensors in a valid context |
| `cuda_context_mismatch` | Paddle tensors on one GPU belong to different contexts; create model state and inputs in the same context |

Torch-private capability probes remain in `_utils/_torch_compat.py`.
`tl.report.doctor()` includes `HAS_FLEX_GEMM`; a missing or changed native
FlexGEMM callable table disables its identity shim and reports that capability.
`HAS_FUNCOL_BATCHED_WAIT_INTERPOSITION` reports availability of list-valued
distributed waits independently of scalar completion tracking.

## SplitFleet

The public `ComputeDAG`, `GraphProfile`, `GraphPartition`, analysis, training
and repartition interfaces remain available. A SplitFleet installation with an
exact 2.x version admission check must explicitly admit this tested release.
Historical pickled metadata fixtures that name old internal modules need an
offline migration; importing legacy modules into TorchLens is not required.
