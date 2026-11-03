# Split runtime acceptance

The current [training DAG and SGD FP64 report](split_training_dag_fp64.md) is the
single source for measured model/device results and repository checks.

## Validation protocol

`benchmarks/split_graph_partition.py` uses plain SGD with lr=0.01, FP64 by default,
and fixed atol=1e-8/rtol=1e-7. Integer/boolean values compare exactly; nonfinite
values, missing gradients, shape and dtype mismatches fail acceptance.

Each training partition kind runs B=1/2/4/8 across A/B/C/A recuts on one captured
graph. Sequential and genuinely noncontiguous DAG results are independent. The
full-model reference checks output, loss, input/parameter/connected-boundary
gradients, updated parameters, current buffers and optimizer bindings. Eval
rows retain their separate dynamic-batch forward checks.

Plain SGD carries no momentum history. A recut binds the current execution
parameters and preserves the same parameter inventory. Model runtime support
remains optimizer-agnostic and preserves the caller's native dtype.

ResNet18 retains BatchNorm and inplace ReLU; YOLO uses its official training
structure and detection criterion; the small DistilBERT fixture retains active
Train-mode Dropout; RF-DETR uses a trusted checkpoint, spatial mask and valid
annotations. The criterion-only floating factory scope follows output precision
and restores the previous default.

## Results and refusal boundaries

Only executed, successful checks count as Passed. Unsupported, skipped, failed
and unrequested capabilities remain distinct. `--strict` writes the complete
JSON report and exits nonzero for every status except Passed.

CPU/CPU, CUDA/CUDA, CPU/CUDA and CUDA/CPU are separate placements. A single GPU
cannot establish two-GPU migration. Dynamic batch probes and captured paths are
finite evidence; unknown aliases, unsupported effects, native call splitting and
reverse dependencies retain their typed refusals.

See [runtime semantics](split_native_runtime.md), [repair mechanisms](split_runtime_repairs.md)
and the [SGD report](split_training_dag_fp64.md) for details and measured coverage.

```bash
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --dtype float64 --prefix-device cpu --suffix-device cuda --strict --output yolo-sgd.json
python -m pytest tests/split -q -rs --tb=short
```
