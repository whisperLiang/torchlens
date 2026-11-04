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

## Backend environments

Keep CUDA 12 and CUDA 13 GPU extras in separate environments. In particular,
`nvidia-cudnn-cu12` and `nvidia-cudnn-cu13` install into the same
`nvidia/cudnn/lib` directory: both package metadata records can remain present
while one set of binaries replaces the other. Record the library actually
loaded by the backend, such as `torch.backends.cudnn.version()`, alongside its
declared dependencies before attributing a native kernel failure to Split.

The MLX 0.26 CPU JIT invokes the `g++` found on `PATH`. Use a compiler compatible
with that wheel's generated preamble; compiler failures do not attest Split
correctness. Keep native compilation enabled during its validation.

### Recovering native backend environments

First run the failing model directly, without TorchLens. A native forward that
aborts cannot establish a Split replay failure. For example, a mixed Torch 2.10
CUDA 12 / cuDNN CUDA 13 installation can abort a MobileNetV2 convolution with
`Invalid handle. Cannot load symbol cublasLtCreate`. In the reproduced environment,
Torch required cuDNN 9.10.2.21 but `torch.backends.cudnn.version()` returned 91300.
Both cuDNN distributions owned the same `libcudnn*.so.9` paths. Installed package
versions alone therefore did not describe the binaries being executed.

Create a separate environment to keep each CUDA runtime's binaries together.
This Torch 2.10 recovery recipe preserves the original environment:

```bash
uv venv .venvs/torch --python python3.12
uv pip install --python .venvs/torch/bin/python -e '.[dev,test,tabular,detection]' \
  'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0'
.venvs/torch/bin/python -c 'import torch; print(torch.__version__, torch.backends.cudnn.version())'
.venvs/torch/bin/python -m pytest tests/split/test_split_real_models.py \
  -k torchvision_extra -x --tb=short
```

Keep JAX, TensorFlow and Paddle GPU extras in their own environments. When the
environment is for another backend, install CPU Torch first so its mandatory
Torch dependency does not bring a competing CUDA runtime. Run that backend's
tests with its own interpreter. Do not add all GPU extras to the Torch environment.

For the Linux MLX 0.26 CPU wheel, use GCC 12 for native JIT compilation when
GCC 13 rejects the generated `_Float*` typedefs:

```bash
uv pip install --python .venvs/torch/bin/python 'mlx==0.26.3'
mkdir -p .venvs/torch/mlx-compiler
ln -sf /usr/bin/g++-12 .venvs/torch/mlx-compiler/g++
PATH="$PWD/.venvs/torch/mlx-compiler:$PATH" .venvs/torch/bin/python -m pytest \
  tests/split/test_split_mlx_training.py tests/split/test_split_mlx_placement.py \
  tests/split/test_split_mlx_microbatch.py tests/split/test_split_mlx_state_io.py \
  tests/split/test_split_mlx_review_regressions.py -x --tb=short
```

The compiler override applies only to that command. Native compilation and all
numerical checks stay enabled; CPU MLX results do not attest GPU placement.

See [runtime semantics](split_native_runtime.md), [repair mechanisms](split_runtime_repairs.md)
and the [SGD report](split_training_dag_fp64.md) for details and measured coverage.

```bash
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --dtype float64 --prefix-device cpu --suffix-device cuda --strict --output yolo-sgd.json
python -m pytest tests/split -q -rs --tb=short
```
