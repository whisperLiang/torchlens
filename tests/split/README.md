# Split runtime acceptance

Floating functional acceptance uses FP64 where supported; production replay
keeps the caller's dtype. The opt-in `fp64_functional` fixture supplies double
test construction for the partition/mutation/buffer/workflow oracles and restores
the prior default afterward. Explicit dtype contracts, optional backend
compatibility tests and performance workloads retain their declared precision.
`test_partition_effect_dependencies.py` covers native independent BatchNorm
branches, inplace activations, A/B/C/A buffer and SGD parameter bindings, RAW/WAR/WAW
refusals, later buffer alias replacement, explicit Generator ordering and
float16/bfloat16/float32/float64 passthrough. `test_split_compute_dag.py` covers
call projection, multiple outputs, source filtering, data/effect dependencies,
local-write obligations and conversion into the same native planner.

The isolated model driver uses plain SGD (`lr=0.01`), defaults to FP64, and
checks integer/boolean values exactly:

```bash
python benchmarks/split_graph_partition.py --model resnet18 --dump-dag graph.json --output sequential.json
python benchmarks/split_graph_partition.py --model resnet18 --model-mode train --partition-kind noncontiguous_dag --partitions workloads.json --strict --output training.json
```

External algorithms supply `workloads.json`: map `"sequential"` and/or
`"noncontiguous_dag"` to lists of prefix Call ID lists from the exported DAG.
Three distinct workloads support the A/B/C/A training check. The default driver
measures three fixed sequential workloads. It reports native analysis refusals
without searching for another cut. DAG construction, analysis, materialization,
forward/backward and repartition are measured independently; no candidate counts
are used as correctness evidence. See the
[native workflow](../../docs/reference/split_native_runtime.md).

The comparison budget is fixed at `atol=1e-8`, `rtol=1e-7`. Tests reject
out-of-budget floating values, dtype/shape mismatches, categorical differences,
nonfinite values and missing gradients.

The transport regressions include exact dynamic split operand lowering
(`test_split_dynamic_operands.py`), persistent mutable buffers and prefix autograd
storage (`test_split_partition_buffer_gradients.py`), and native stochastic replay
across real CPU/CUDA devices (`test_split_rng_devices.py`). They compare values,
gradients and effective state across recuts; hardware skips are not passes.

All workflow tests select native calls with `compute_dag().to_partition(call_ids)`.
Planner/refusal tests also construct `GraphPartition` directly from canonical compute IDs.
True DAG tests retain interleaved branch sets. There is no deferred request or point-selector
compatibility layer.

`dag_helpers.py` is an external, test-only partition oracle. Small graphs enumerate all
assignments under the current DAG's directed data/effect dependencies and co-location groups.
Real-model matrices execute the 100 native-legal partitions with the smallest forward boundary
transfer, or every legal partition if fewer than 100 exist. A bounded minimum-cut algorithm
searches disjoint assignment regions in exact cost order; it does not restrict the search to
linear prefixes. Native `analyze()` is the final legality authority, including conditional
mutation refusals. Independent small-graph exhaustive oracles check completeness and ranking.

Costs come from `graph.graph_profile()` at the graph's canonical capture batch. Each frontier
tensor value is charged once, including fan-out, fixed input/buffer provenance and returned
outputs. Effects carry no communication charge. Unknown tensor sizes refuse ranked search;
when the entire small legal set is tested, ranking is unnecessary. The production package
contains no partition candidate generator.

`test_partition_benchmark.py` exercises strict/diagnostic subprocess exits, saved
JSON, missing trusted weights, native reference observation and current buffers.
The acceptance driver trains A/B/C/A with SGD across B=1/2/4/8, checking
optimizer bindings and rebinding the current execution parameters explicitly.
CPU, CPU→CUDA, CUDA→CPU, same-GPU and two-GPU cases are separate; missing physical
devices skip. BatchNorm state regressions cover compact/diagnostic products,
connected training, sequential microbatches and state preservation after recuts.

Duration-budget enforcement has been removed. Test markers
still select suites; elapsed time does not change a correctness verdict.

Run the repository quality gates with the versions declared in `[project.optional-dependencies.dev]`:
in particular Ruff 0.15.4 and mypy 2.1.0. The full-suite oracle tests also require
a committed environment fingerprint; see [oracle environments](../oracle_environments.md).
Run with `TORCHLENS_ORACLE_ENFORCE=1` in a supported environment. When copying
a Windows checkout to Linux, preserve Git's original LF
bytes for generated artifacts; CRLF copies fail byte-identity checks even when
the renderer output is unchanged. Never regenerate oracle expectations merely
to make a new environment pass.

The split suite includes optional native backends, pretrained model checks, and
native DAG partition matrices. Missing GPU builds are environment failures in the
cross-device real-model tests; a CPU-only JAX or Paddle installation does not cover
them. TensorFlow GPU XLA also needs the CUDA compiler package's
`nvvm/libdevice/libdevice.10.bc`, in addition to runtime libraries. Install a compiler
package matching the CUDA major version shared by the installed frameworks.

Launch pytest with NVIDIA library directories on `LD_LIBRARY_PATH` **before the
Python process starts**. Setting this only for model subprocesses leaves parent
TensorFlow tests unable to discover their GPU. When CUDA comes from Python wheels,
set `XLA_FLAGS=--xla_gpu_cuda_data_dir=.../site-packages/nvidia/cuda_nvcc` to the
installed compiler directory. Keep backend CUDA package versions compatible;
installing several framework GPU extras together can request conflicting versions.
For native crashes, conflicting cuDNN binaries, or MLX CPU compiler errors, follow
the [backend recovery recipes](../../docs/reference/split_runtime_acceptance.md#recovering-native-backend-environments)
before rerunning the same numerical tests.

For a reproducible model-package setup with Torch 2.10, install these versions
alongside the required native backend runtimes:

```bash
python -m pip install 'rfdetr==1.8.3' 'ultralytics==8.4.152' 'keras-cv==0.9.0'
```

RF-DETR 1.8.3 follows the declared `>=1.8,<1.9` dependency range. Pretrained
weight digests remain checked; raw operation counts are not acceptance oracles.

Enable all model matrices and allocated-memory comparisons, including rare tests:

```bash
export TORCHLENS_REAL_MODEL_TESTS=1 TORCHLENS_REAL_MODEL_STRICT=1
export TORCHLENS_CUDA_MEMORY_TESTS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
export TF_FORCE_GPU_ALLOW_GROWTH=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
python -m pytest -o addopts= tests/split -q -rs --tb=short --randomly-seed=20260920
```

Run optional backends in their own compatible environments. On a shared GPU, schedule
memory-heavy models sequentially in fresh pytest processes:

```bash
python -m pytest -o addopts= tests/split -m slow \
  -k 'not yolov8n_smallest_dag_partitions and not rfdetr_smallest_dag_partitions' -v -rs
python -m pytest -o addopts= tests/split/test_split_real_models.py::test_yolov8n_smallest_dag_partitions_cross_batch_and_device -v -rs
python -m pytest -o addopts= tests/split/test_split_real_models.py::test_rfdetr_smallest_dag_partitions_cross_batch_and_device -v -rs
python -m pytest -o addopts= tests/split -m 'not slow' -v -rs
```

The YOLOv8 and RF-DETR matrices assert 100 tested native DAG partitions. Both check CPU
and CUDA replay at B=1/2/3 and transfer boundaries in both directions at B=1/3. Independent
training cases retain gradient, optimizer, mutable-state and recut assertions. The first
pretrained run may download model weights. Optional `TORCHLENS_RFDETR_PROGRESS=/path/prefix`
appends completed-partition progress to `/path/prefix.dag.log`.

Fresh model subprocesses avoid retained parent CUDA contexts. Cleanup clears existing
Torch caches without creating a new context. Provide sufficient free GPU memory; capture
budget refusals are failures in strict runs.
