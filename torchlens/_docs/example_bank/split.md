# Native Split Runtime

### split-prepare-and-execute
Execute a concrete dependency-closed partition using the public request API.

- covers: `tl.prepare`, `tl.split`, `tl.SplitRequest`, `tl.SplitFeatures`, `tl.SplitRuntime`, `tl.SplitGraphIR`, `tl.ReplayBoundary`, `tl.BoundarySchema`
- tasks: split, replay
- fixture: toy

```python
graph = tl.split.prepare_graph(model, x)
partition = graph.compute_dag().to_partition([])
request = tl.SplitRequest(partition=partition, features=tl.SplitFeatures())
runtime = tl.prepare(model, x, request)
assert isinstance(runtime, tl.SplitRuntime)
assert isinstance(runtime.graph_ir, tl.SplitGraphIR)
boundary = runtime.run_prefix(x)
assert isinstance(boundary, tl.ReplayBoundary)
assert all(isinstance(item, tl.BoundarySchema) for item in runtime.boundary_schema)
torch.testing.assert_close(runtime.run_suffix(boundary), model(x))
print("native split replay passed")
# Output:
# native split replay passed
```

### split-model-profile-and-verification
Describe an externally selected model and inspect a native verification status.

- covers: `tl.SplitModelProfile`, `tl.SplitVerificationStatus`
- tasks: split, profile
- fixture: toy

```python
profile = tl.SplitModelProfile(id="toy", backend="torch", family="mlp")
assert profile.loader is None
print(profile.id, profile.backend, tl.SplitVerificationStatus.EXACT.value)
# Output:
# toy torch exact
```
