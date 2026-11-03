# YOLO CPU/CUDA tolerance revalidation

This follow-up records the explicitly requested comparison profile
`cross-device-fp64-v1`: `atol=1e-7`, `rtol=1e-7`. It applies only to FP64
CPU/CUDA segment pairs and is selected with `--tolerance-profile`.

The default `strict` profile retains `atol=1e-8`, `rtol=1e-7`. The earlier
[strict FP64 matrix](split_training_dag_fp64.md) and its failed rows keep their
original results. A new-profile pass does not relabel a historical experiment.

Only the isolated benchmark's floating comparisons change. Shape/dtype,
integer/boolean values, gradient presence, and finite-value checks remain
mandatory. The scope restores the preceding profile on normal and exceptional
exit. Capture, replay, model forward, BatchNorm, inplace operations, official
loss, gradients and optimizer formulas are unchanged.

The real-device revalidation uses the prior Orin environment, FP64 YOLOv8n
at 160x160, independent full Native controls, and both Sequential and
noncontiguous DAG partitions. Train checks retain B=1/2/4/8, SGD, Momentum,
Adam and A/B/C/A repartition. Native controls record numerical violations
without aborting later steps; any such violation leaves the control Failed.
Structural, categorical and nonfinite failures still abort the control.

## Measured result

The larger absolute tolerance moves both Split directions to the final Adam
batch, but both remain **Failed**. The independent full-Native controls also
fail this budget. This experiment does not establish a generally sufficient
CPU/CUDA tolerance or a passed cross-device Adam trajectory.

| Split placement | Sequential completed checks / requested | DAG completed checks / requested | First failed step |
| --- | ---: | ---: | --- |
| CPU / CUDA | 11 / 12 | 11 / 12 | Adam B=8, boundary value |
| CUDA / CPU | 11 / 12 | 11 / 12 | Adam B=8, boundary value |

The first failing boundary comparison has maximum absolute differences of
2.072e-6 / 1.372e-6 for CPU/CUDA Sequential/DAG and 5.860e-7 / 2.823e-7 for
CUDA/CPU. The corresponding historical strict runs completed 9 and 10 checks
per kind, respectively. Check counts are fully validated records; a failing
step can have executed some operations without completing all its checks.

All four full-Native controls execute all 12 steps. Their Failed status and
individual numeric violations are retained while subsequent steps continue:

| Native reference / actual | Deterministic | Checks passed / steps executed | Maximum forward difference | Maximum parameter-gradient difference |
| --- | --- | ---: | ---: | ---: |
| CPU / CUDA | False | 11 / 12 | 8.667e-7 | 4.278e-7 |
| CPU / CUDA | True | 11 / 12 | 9.191e-7 | 5.331e-7 |
| CUDA / CPU | False | 10 / 12 | 5.098e-6 | 1.777e-5 |
| CUDA / CPU | True | 10 / 12 | 4.513e-6 | 1.378e-5 |

These are maximum absolute errors, not per-element acceptance bounds: the
actual check remains `abs(a-b) <= atol + rtol * abs(b)`. No structural,
categorical, dtype or nonfinite error was recorded in these Native controls.
Each optimizer starts from the same initial weights; Adam's trajectory is
independent of the preceding SGD/Momentum trajectories. All eight SGD/Momentum
steps pass in every Native and Split case.

The same-GPU deterministic Split control keeps the original strict profile
and passes 12/12 checks per kind (24/24 in total). Runtime execution code is
unchanged; the new comparison policy lives only in the isolated benchmark.

The hardware environment remains Python 3.10.12, Torch 2.11.0, CUDA 12.6,
Ultralytics 8.4.152 and one Orin GPU, with one CPU thread and seed 42. cuDNN
benchmarking is disabled, cuBLAS workspace is `:4096:8`, and TF32 is disabled
by the process environment. The full-Native controls test determinism off/on;
the requested mixed-device Split rows use the default algorithm policy.

The numerical source packet has normalized 166-file manifest SHA256
`40b0f5141eb7b4ba149e68b0171d9f1d297b35bb1cf0389709daf590c2dd3d6a`.
A subsequent test-only style cleanup combines nested context managers without
changing the benchmark. Its floating policy and execution source are identical
to the measured packet. Original strict artifacts are not overwritten.

The benchmark/test change is commit `05a4933c8`. The final normalized 166-file
repository manifest has SHA256
`62a1607fcc24bad3d43f8eb9127e7308db45529bf3d79531c73bc9f54dab9cf5`.
The benchmark itself matches the numerical packet byte-for-byte after newline
normalization; only the test style cleanup differs. All production Split source
hashes match the preceding implementation.

## Regression checks

| Check | Result |
| --- | --- |
| Complete benchmark suite, local | 32 passed, 2 real-CUDA skips |
| Complete benchmark suite, Orin | 34 passed |
| Focused Orin CPU/CUDA regressions | 155 passed, 6 two-GPU skips |
| Complete local Split suite | 542 passed, 233 hardware/optional skips |
| Skip, complexity and file-size gates | 26 passed |
| Repository format / Ruff | Passed |
| mypy `torchlens/` | Passed, 613 source files |
| Windows smoke | 5,745 passed, 178 skipped, 1 xfail; 10,570 deselected |

Two manual guard mutations are detected by the new tests: leaking the selected
profile past a nested/exceptional scope and applying approximate comparison to
large categorical integers. These expected mutant failures are distinct from
the passing suite. The unavailable dedicated CRAP/duplication/mutator tools
are not represented as a completed formal campaign.

## Interpretation

The earlier first violations near 1e-8 understated later trajectory differences.
The complete Native runs expose parameter-gradient differences up to about
1.8e-5 by Adam B=8, so increasing only the absolute budget to 1e-7 is insufficient
for this workload. The experiment does not identify the first divergent kernel
or rule out additional Split contributions. Further threshold choices need a
separate error budget and held-out calibration; these failures are not relabelled
as passes and the threshold is not increased automatically.

Reproduce the two requested placements in separate processes:

```bash
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --dtype float64 --device cpu --prefix-device cpu --suffix-device cuda --tolerance-profile cross-device-fp64-v1 --strict --output yolo-cpu-cuda-v1.json
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --dtype float64 --device cuda --prefix-device cuda --suffix-device cpu --tolerance-profile cross-device-fp64-v1 --strict --output yolo-cuda-cpu-v1.json
```

The same-GPU control keeps the strict profile and adds `--deterministic`.
This is a finite, fixed-seed functional experiment; broader inputs, hardware
and longer trajectories require their own validation.
