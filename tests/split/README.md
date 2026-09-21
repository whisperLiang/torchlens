# Split runtime acceptance

The split suite includes optional native backends, pretrained model checks, and
exhaustive boundary matrices. Missing GPU builds are environment failures in the
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

Enable the complete suite, including all YOLOv8 and RF-DETR boundaries and allocated
memory comparisons:

```bash
export TORCHLENS_REAL_MODEL_TESTS=1 TORCHLENS_REAL_MODEL_STRICT=1
export TORCHLENS_YOLOV8_EXHAUSTIVE=1 TORCHLENS_RFDETR_EXHAUSTIVE=1
export TORCHLENS_CUDA_MEMORY_TESTS=1
export TORCHLENS_YOLOV8_PARTITIONS=1 TORCHLENS_YOLOV8_PARTITION=0
export TORCHLENS_RFDETR_PARTITIONS=1 TORCHLENS_RFDETR_PARTITION=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
export TF_FORCE_GPU_ALLOW_GROWTH=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
python -m pytest tests/split -q -rs --tb=short --randomly-seed=20260920
```

The exhaustive tests can each take up to two hours. Their partition controls are
for separately scheduled coverage; `PARTITIONS=1` runs every boundary in one test.
The first pretrained run may download model weights.

On a shared GPU with limited free memory, run these selections **sequentially**,
in place of the final command above:

```bash
python -m pytest tests/split -m slow \
  -k 'not yolov8n_all_split_nodes and not rfdetr_all_split_nodes' \
  -v -rs --tb=short --randomly-seed=20260920
python -m pytest tests/split/test_split_real_models.py::test_yolov8n_all_split_nodes_cross_batch_and_device \
  -v -rs --tb=short --randomly-seed=20260920
python -m pytest tests/split/test_split_real_models.py::test_rfdetr_all_split_nodes_cross_batch_and_device \
  -v -rs --tb=short --randomly-seed=20260920
python -m pytest tests/split -m 'not slow' -v -rs --tb=short --randomly-seed=20260920
```

These disjoint selections cover every split test exactly once. The exhaustive
models get fresh parent processes, so earlier Torch tests cannot retain CUDA
contexts while those model subprocesses capture. The helper clears existing parent
Torch caches but never initializes a new parent CUDA context merely to clean it up.
A fresh process cannot compensate for insufficient free device memory: RF-DETR's
GPU capture can still refuse its activation budget on a heavily occupied device.
Provide more free GPU memory rather than treating that refusal as a skipped check.

The YOLOv8 inventory explicitly admits Ultralytics 8.4.152's 285 compute nodes
(570 before/after boundaries), including nine `new_full` nodes. This official
anchor builder adds six allocation nodes over the legacy 279-node profile. Other
versions retain the legacy assertion until their graph change is reviewed; the
test does not silently accept an arbitrary node count.
