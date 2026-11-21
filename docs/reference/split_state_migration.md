# Split optimizer state migration

A split model can change its DAG partition between training steps. The
optimizer migration API plans the transition, exports independent CPU state,
and builds fresh native optimizers for the new parameter owners. Both PyTorch
and TensorFlow use the same three methods on `SplitRuntime`.

| Method | Result |
| --- | --- |
| `plan_optimizer_state_migration(partition, optimizers=...)` | An `OptimizerStateMigrationPlan` with a decision for each trainable parameter and any refusal reasons. |
| `export_optimizer_state(optimizers=..., plan=...)` | An `OptimizerStateSnapshot` bound to the source and target partitions. |
| `restore_optimizer_state(snapshot, optimizer_factories=...)` | An `OptimizerStateMigrationResult` with new native optimizers and counts of restored and initialized slots. |

Call these methods between completed training steps. Supply one optimizer for
each side with trainable parameters. Omit an empty owner. Pass the returned
optimizers to `train_suffix()` and `backward_prefix()` to continue native
training.

## Parameter IDs and model state

`parameter_ownership()` describes named state, its consuming nodes, and its
owners. A parameter ID begins with `parameter:` and uses a canonical path
such as `parameter:encoder.weight` in PyTorch or
`parameter:dense/kernel` in Keras. Tied aliases refer to one ID. In
TensorFlow, the live resource behind a Keras or TensorFlow variable joins its
captured reads to the current training bindings. This also handles cuts inside
a layer. A captured variable without a provable live read, a replaced
resource, or an unresolved name collision is refused.

`parameter_bindings()` returns the effective native tensors for those IDs.
`state_bindings()` also includes used buffers and the Torch replay literals.
`plan_state_migration()` describes movement of model state, and
`repartition()` inherits that state. Torch uses `SegmentState` to track its
effective values and replicas. An optimizer snapshot carries optimizer
history separately. Restore it against a runtime whose model weights already
match the export step.

## Plans and snapshots

`OptimizerStateMigrationPlan.supported` is true if the plan can proceed.
Read `unsupported_reasons` when it is false. Each
`OptimizerStateMigration` records the `state_id`, `source_optimizer`,
`target_optimizer`, `action`, and `slot_names`. Actions are `keep` for the
same owner, `transfer` for a new owner, `initialize` for history the native
optimizer has yet to create, and `unsupported` for a refused plan. A change to
the source optimizer's state or configuration after planning requires a new
plan.

A version-1 `OptimizerStateSnapshot` stores its `backend`, `graph_id`,
`source_partition`, `target_partition`, and `entries`. The `optimizers`,
`adapter_names`, and `parameter_specs` mappings specify the target optimizer
configuration, global state, and parameter inventory. Each `OptimizerState`
holds native `optimizer_type`, portable `config`, `global_state`,
`expected_slots`, and actual `slots` by stable parameter ID. Each
`OptimizerStateTensor` carries the state ID, slot name, shape, dtype, immutable
CPU NumPy payload, and a SHA256 checksum. The snapshot checksum also covers the
manifest. The snapshot is an in-memory handoff. Manage any checkpoint file
format independently.

`OptimizerStateMigrationResult.optimizers` maps nonempty owners to their
fresh native optimizers. `restored_parameters` lists the IDs whose histories
were retained. `restored_slots` counts per-parameter tensors restored.
`initialized_slots` counts expected slots delegated to native initialization.
`migration_summary` counts decisions by action.

## Change a cut during training

Both examples train one step, take an optimizer snapshot for the new cut,
and continue training with the restored optimizers.

### TensorFlow

This FP64 example begins with the entire model on the suffix. It moves the
first Dense layer and its Adam history to a training prefix.

```python
import tensorflow as tf
from torchlens.split import SplitFeatures, prepare_graph

tf.keras.utils.set_random_seed(7)
model = tf.keras.Sequential([
    tf.keras.Input((4,), dtype="float64"),
    tf.keras.layers.Dense(5, activation="relu", dtype="float64", name="stem"),
    tf.keras.layers.Dense(3, dtype="float64", name="head"),
])
x, y = tf.ones((2, 4), tf.float64), tf.zeros((2, 3), tf.float64)
graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
dag = graph.compute_dag()
runtime = graph.materialize(graph.analyze(dag.to_partition(())))
optimizers = {"suffix": tf.keras.optimizers.Adam(0.01)}
runtime.train_suffix(runtime.run_training_prefix(x), y, optimizer=optimizers["suffix"])

# Read kernel, multiply, read bias, add bias, and activate the first layer.
cut = dag.to_partition(call.call_id for call in dag.nodes[:5])
plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
assert plan.supported, plan.unsupported_reasons
snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
runtime = runtime.repartition(cut)
optimizers = runtime.restore_optimizer_state(snapshot).optimizers
boundary = runtime.run_training_prefix(x)
_, grads = runtime.train_suffix(boundary, y, optimizer=optimizers["suffix"])
runtime.backward_prefix(boundary, grads, optimizer=optimizers["prefix"])
assert {int(opt.iterations) for opt in optimizers.values()} == {2}
```

### PyTorch

Moving the Linear layer also moves its AdamW moments and per-parameter steps.

```python
import torch
from torch import nn
from torchlens.split import SplitFeatures, prepare_graph

torch.manual_seed(7)
model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3)).double()
x, y = torch.ones(2, 4, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64)
graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
dag = graph.compute_dag()
runtime = graph.materialize(graph.analyze(dag.to_partition(())))
optimizers = {"suffix": torch.optim.AdamW(model.parameters(), lr=0.01)}
runtime.train_suffix(runtime.run_training_prefix(x), y, optimizer=optimizers["suffix"])

cut = dag.to_partition(call.call_id for call in dag.nodes[:2])
plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
assert plan.supported, plan.unsupported_reasons
snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
runtime = runtime.repartition(cut)
optimizers = runtime.restore_optimizer_state(snapshot).optimizers
boundary = runtime.run_training_prefix(x)
_, grads = runtime.train_suffix(boundary, y, optimizer=optimizers["suffix"])
runtime.backward_prefix(boundary, grads, optimizer=optimizers["prefix"])
assert all(int(state["step"]) == 2 for opt in optimizers.values() for state in opt.state.values())
```

## Supported optimizers and refusals

| Backend | Native classes | Migrated state |
| --- | --- | --- |
| TensorFlow with Keras 3 | SGD, SGD with momentum, Adam, Adam with AMSGrad, AdamW | Iterations, mutable learning rate, momentum, velocity, and the AMSGrad maximum where configured. Built-in stateless learning-rate schedule configuration is retained. |
| PyTorch | SGD, SGD with momentum, Adam, AdamW | Parameter groups and their effective hyperparameters, momentum buffers, moments, AMSGrad maxima, and per-parameter steps. |

Keras optimizers contributing to the same new owner must have identical update
configuration, iterations, and mutable learning rates. Torch preserves
individual parameter-group rates and requires matching constructor defaults
when joining histories. A trainable parameter used by both segments has
ambiguous independent optimizer ownership and is refused. Unsupported Keras
state includes EMA, gradient accumulation, dynamic loss scaling, global
gradient clipping, weight-decay exclusions, and opaque learning-rate
callables. Sparse and differentiable Torch optimizer histories are outside the
dense state contract. External scheduler objects and training-loop state
remain caller-owned. [Keras optimizer API](https://keras.io/api/optimizers/)

On an unsupported plan, inspect `unsupported_reasons`. Export and restore
raise `SplitUnsupportedError` for unsupported or stale plans, incompatible
targets, and corrupt state. Restore validates the schema, graph, partition,
and parameter/slot inventory before it constructs optimizers. Native state
writes apply to fresh optimizer state. Publish the returned optimizer mapping
as a unit after restore succeeds.

## Cross-device restore

The payloads store CPU bytes. PyTorch uses the target's effective bindings,
including model replicas made by the existing placement path. TensorFlow
uses the devices of the target runtime's existing variables. To move a
TensorFlow workload between CPU and GPU, prepare the same native graph on the
destination device with matching model values and choose the corresponding
partition. Restore verifies the target graph and state descriptors. A
TensorFlow runtime keeps its existing replay placement policy.

## Adapter extension

Register an `OptimizerStateAdapter` through
`torchlens.split.optimizers.register_optimizer_adapter`. Retrieve an adapter
for a native optimizer with `get_optimizer_adapter`. An adapter's unique
`name` identifies it in a snapshot. Implement `supports`, `export_state`,
`merge_config`, `validate_restore`, `build_optimizer`, and `restore_state`.
Use `OptimizerStateTensor.from_array` to encode native slots and
`OptimizerState` to declare their complete inventory. Duplicate adapter names
or ambiguous matches raise `SplitUnsupportedError`.

Built-in classes can be reconstructed automatically. A custom adapter uses
an explicit `OptimizerFactory` supplied by the caller. The factory receives
two positional arguments, a mapping from stable ID to effective target
variable and a plain copy of the verified optimizer config. It must return a
fresh optimizer without modifying the live variables or the source optimizers.
