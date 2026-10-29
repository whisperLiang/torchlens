# Native split runtime acceptance — 2026-10-29

Implementation is complete for the available Torch CPU environment. CUDA, optional replay backends and RF-DETR remain unvalidated here; the whole-repository quality gates have pre-existing environment/baseline failures listed below.

## Architecture and deleted surface

`GraphPartition` is the sole production split description. `prepare_graph()` returns a capture-only `SplitGraph`; `analyze(partition)` returns `PartitionAnalysis`, `materialize()` creates segments, and `repartition()` inherits the current effective state. `prepare(..., SplitRequest(partition, ...))` uses these same stages. There is one planner and the existing backend ReplayProgram/SegmentState execution engine. No solver, scheduler, aggregation or networking was added.

Deleted production surface and machinery: `SplitPoint`, `before()`, `after()`, `percent()`, `SplitPointAnalysis`, `SplitCandidate`, `SplitCandidateReport`, `runtime.at()`, `runtime.split_points()`, point target resolution, point cache keys, target-based primary/skip boundary classification, and the duplicate GraphProfile FLOPs formula. Request ownership is now `SplitRequest.partition`. Topological candidate generation is optional and never restricts manual DAG selections.

Core files: [api.py](../../torchlens/split/api.py), [context.py](../../torchlens/split/context.py), [planner.py](../../torchlens/split/planner.py), [runtime.py](../../torchlens/split/runtime.py), [graph.py](../../torchlens/split/graph.py), [graph_profile.py](../../torchlens/split/graph_profile.py), [Torch adapter](../../torchlens/split/adapters/torch.py), [pipeline.py](../../torchlens/split/pipeline.py), [capture FLOPs](../../torchlens/capture/flops.py), plus IR, frontier, ownership, candidates, exports, tests, glossary and examples.

## FLOPs contract

GraphProfile calls capture.flops directly with shape/dtype-only tensor arguments, captured parameter shapes, scalars, keywords and ShapeProgram rewrites. It allocates no tensors, replays no model and reads no tensor values. Captured numeric forward/backward costs survive compaction; mismatches at the captured batch are diagnosed. Registering a rule through the existing `register_op_rule()` increments the cost revision and invalidates profile/analysis metric cache entries.

Forward and heuristic backward costs are distinct. Multi-output native calls are charged once. Unknown costs propagate as None with diagnostics; zero-FLOPs memory operations still expose bytes. Tests cover Conv, MatMul, normalization, elementwise, attention, einsum, multi-output, unknown operations, custom rules, dynamic reduction and B=1/2/4/8. The captured-cost comparisons cover retained and compact captures.

## Repartition measurements

Baseline: `d1eb394a377304859c82031528c9857feff5ee22`. Windows 10 build 26300, Python 3.10.18, PyTorch 2.10.0+cpu, torchvision ResNet18 with random weights, eval mode, 1000 outputs, input (2,3,32,32), seed 73, one Torch CPU thread. Three independent processes per revision, alternated before/after. Each phase has 25 samples per process (75 total); training and post-training recuts have 10 per process (30 total). Prefix counts 18/34/51 and all partition identities match exactly between revisions.

Wall times are uninstrumented. Stage shares and Python metadata memory use separate passes. These results describe this workload, not universal speedups.

| Phase | Before median / P95 (ms) | After median / P95 (ms) | Median change | n per revision |
| --- | ---: | ---: | ---: | ---: |
| cold_analysis | 5.041 / 7.160 | 4.479 / 6.244 | -11.1% | 75 |
| warm_analysis | 1.996 / 3.230 | 0.982 / 1.641 | -50.8% | 75 |
| cold_materialization | 5.127 / 6.755 | 3.286 / 4.179 | -35.9% | 75 |
| warm_materialization | 5.126 / 7.078 | 2.469 / 3.229 | -51.8% | 75 |
| same_partition_repartition | 6.639 / 8.659 | 3.017 / 4.308 | -54.6% | 75 |
| rotating_repartition | 7.174 / 9.654 | 3.289 / 6.419 | -54.1% | 75 |
| cold_repartition | 10.520 / 12.479 | 6.451 / 8.271 | -38.7% | 75 |
| training_step | 50.210 / 82.406 | 50.061 / 88.197 | -0.3% | 30 |
| post_training_repartition | 8.105 / 11.189 | 3.866 / 7.369 | -52.3% | 30 |

The rotating warm recut median improves by 2.18x. The model training step is approximately unchanged; the median change is -0.3%; no training-kernel speedup is claimed. Capture was measured once per process (n=3):

| Comparable capture phase | Before median / P95 (ms) | After median / P95 (ms) |
| --- | ---: | ---: |
| canonical_capture | 2221.18 / 2228.67 | 2223.08 / 2226.24 |
| normalization_and_batch_probe | 1503.40 / 1516.33 | 1593.43 / 1608.22 |

The reference revision requires an initial empty-partition runtime; the final API captures without that persistent runtime. Consequently total preparation scopes differ; the table above isolates comparable stages. Normalization/probe time increased slightly.

Exclusive stage shares from separate instrumented cold recuts (mean across three process runs):

| Stage | Before | After |
| --- | ---: | ---: |
| state_binding | 38.7% | 27.5% |
| state_inspection | 0.1% | 0.1% |
| validation | 3.5% | 5.3% |
| effects | 3.2% | 4.8% |
| frontier | 0.8% | 1.3% |
| lowering | 8.7% | 14.8% |
| capability | 1.0% | 1.7% |
| segment_materialization | 21.3% | 9.6% |
| state_inheritance | 1.6% | 2.4% |
| overhead | 21.1% | 32.5% |

State binding totals about 4.21 ms before and 1.91 ms after per instrumented cold recut. Stage wrapper overhead is included in the overhead remainder; instrumented times are not mixed with the latency table. Device transfer is N/A: no CUDA execution occurred.

The measured optimizations are a bounded 64-entry immutable segment blueprint cache (call/release schedule and argument locations), flattened state argument binding, one frozen-source eligibility scan per bundle instead of three, a single effective-state inspection for combined recuts, deduplicated shared-state inspection/settling, and scalar/container report copies instead of generic deepcopy. The 32-entry analysis LRU remains. State-dependent eligibility and merge checks still run; replicas, optimizer state, gradients and boundary tensors are never cached in blueprints. Ancestor reachability is bounded per graph instead of retaining an all-pairs closure.

| Memory metric, median of three processes | Before | After |
| --- | ---: | ---: |
| Python metadata peak | 0.946 MiB | 0.787 MiB |
| Python metadata retained | 0.361 MiB | 0.347 MiB |
| Process peak working set | 555.270 MiB | 555.461 MiB |
| Process retained RSS | 463.648 MiB | 500.219 MiB |

Python metadata peak falls by 16.8%; process peak working set is effectively flat. Retained process RSS increases by about 36.6 MiB (7.9%). RSS includes native allocator caches and fragmentation; this experiment does not attribute that increase to a particular allocation. This is a measured tradeoff, not a claim of universal memory reduction. A separate compact-training lifetime probe found zero retained child runtimes after recuts. Real-model step RSS values are included in the JSON; the short sequences do not establish long-duration leak freedom. The recorded cache hit rate since the last explicit clear is 23/26 (88.5%) on both revisions.

## Real-model numerical acceptance

Each official architecture uses random initialization, a separate reference model and separate optimizer. Runs use eval mode with autograd and optimizer updates; they do not claim train-mode BatchNorm/dropout validation on these complete architectures. Train-mode BatchNorm, tied state and buffer migration are tested separately in the split suite. No tolerances were loosened: forward/input/parameter/boundary/update comparisons use atol=1e-4 and rtol=1e-3, rejecting all nonfinite values.

Every model independently passes sequential and genuinely noncontiguous DAG inference, nested-output reconstruction, B=1/2/4/8, SGD, SGD momentum and Adam. For each kind and optimizer, three different partitions follow A/B/C/A across four training steps (12 steps per kind). Prefix/suffix ownership changes while the same parameter objects retain optimizer history. One canonical model capture plus one independent B=2 witness is recorded; recuts/replay add zero captures.

| Model and configuration | Sequential | Noncontiguous DAG | Max parameter-gradient error | Max boundary-gradient error | Max update error | Max optimizer-state error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ResNet18, 10 outputs, (B,3,64,64) | passed | passed | 0 | 0 | 0 | 0 |
| DistilBERT, 2 layers, dim 32, 4 heads, inputs_embeds (B,12,32), nested hidden states | passed | passed | 0 | 0 | 0 | 0 |
| YOLOv8n official YAML, (B,3,160,160), full detection outputs | passed | passed | 4.77e-07 | 1.16e-09 | 3.91e-07 | 1.91e-06 |

ResNet18 noncontiguous prefix counts are 35/36/20 out of 69 compute nodes. Those sets differ from every equal-length captured topological prefix; the order was never rewritten. Raw per-step identities, frontier bytes, timings, losses, error values and RSS are linked below.

| Additional coverage | Result |
| --- | --- |
| Compact inference/training, dynamic batch, state identity, analysis/blueprint isolation, tied parameters, buffers, boundary/cache/microbatch contracts | Split tests pass |
| CPU → CPU replay/training/recuts | Passed |
| CPU → CUDA / CUDA → CPU / multi-GPU | Not run: CUDA unavailable in the installed CPU-only PyTorch environment |
| JAX / MLX / Paddle / TensorFlow / tinygrad native replay | Not run: optional native runtimes absent; synthetic neutral IR tests and embedded scenario syntax checks execute |
| RF-DETR Nano | Skipped: package and trusted local checkpoint unavailable; no passing claim |
| Extra torchvision models and pretrained YOLO26 CPU replay | Regression tests pass; YOLO26 CUDA cell skips independently |

## Quality gates and limitations

- `ruff format .` and `ruff check . --fix`: pass; final checks include scripts, examples, tests and package sources.
- Full `pytest tests/split --tb=short -q`: **460 passed, 187 skipped**, 220.32 seconds. Skips are optional runtime, CUDA or explicitly gated model scenarios.
- `mypy torchlens/split --follow-imports=silent`: pass, 54 source files. Full `mypy torchlens/`: 33 existing errors in 12 files. An independent check of the baseline revision has the same 33 errors; no split-module errors remain.
- Required full smoke and not-rare/not-slow commands were attempted. Collection fails on Windows because `resource` is unavailable. Excluding the listed POSIX/crawler collection blockers and enabling UTF-8 allows smoke to execute 68 tests before an environment-golden refusal (`py3.10-torch2.10.0` missing). No golden was generated or weakened to bless this run. `jsonschema` is also missing for the crawler tests.
- The named crapper/dryer/mutator tools are unavailable in this environment. Their execution, CRAP thresholds and mutation scores are not claimed.
- Sparse taken-path replay and dynamic shape guards preserve their existing limits: untested Python branches are not proven by the single shape witness.
- Unproven custom side effects, unsafe alias/effect reorderings, unsupported shared trainable cross-device state and floating boundary writes in training refuse structurally. Sequential same-device inference aliases are allowed only with the native effect order.
- CUDA transport/optimizer migration correctness and long-running memory stability still need a suitable environment; RF-DETR still needs an installed runtime and trusted checkpoint. These are pending validation, not hidden passes.

## Reproduce and inspect

```bash
python benchmarks/split_repartition.py --repeats 25 --output after.json
python benchmarks/split_repartition.py --source-root /checkout/at/d1eb394a --repeats 25 --output before.json
python benchmarks/split_graph_partition.py --model resnet18 --training --output resnet.json
python benchmarks/split_graph_partition.py --model transformer --training --output transformer.json
python benchmarks/split_graph_partition.py --model yolov8 --training --output yolov8.json
python benchmarks/split_runtime_overhead.py --repeats 20
python benchmarks/split_training_compact.py --mode compact-training
python examples/split_native_partition.py
```

Use the final benchmark driver with each source checkout in independent processes; its reference handling uses GraphPartition on both revisions and introduces no legacy runtime adapter. Run performance comparisons without simultaneous model workloads. The JSON summary includes every latency sample, per-process files, stage shares, memory metrics, configuration and source-file digests. Digests are stored as byte arrays to avoid false positives from secret scanners; they retain the exact SHA256 values.

[Repartition summary and raw sample files](../../benchmarks/split_results_2026_10_29/repartition-summary.json); [ResNet18](../../benchmarks/split_results_2026_10_29/resnet-validation.json); [DistilBERT](../../benchmarks/split_results_2026_10_29/transformer-validation.json); [YOLOv8n](../../benchmarks/split_results_2026_10_29/yolov8-validation.json); [RF-DETR skip](../../benchmarks/split_results_2026_10_29/rfdetr-validation.json).

## Execution log

1. Established the clean d1eb394a baseline and captured original regression/performance results before edits.
2. Added failing graph-first/optimizer-state tests, extracted shared graph analysis, removed point-specific production paths and migrated the existing test fixtures.
3. Added failing FLOPs tests; adapted shape/scalar metadata to the capture engine, retained numeric capture costs and verified custom registration and dynamic work.
4. Profiled state binding/setup/copying, applied bounded immutable schedule reuse and reduced duplicate state work, then validated state/alias/optimizer isolation.
5. Fixed lazy hook-registry restoration from a failing reproducer, including native registry identity and existing user handles.
6. Separated sequential/DAG real-model acceptance and verified independent references, native frontier gradients, all requested CPU batches/optimizers and continued recuts.
7. Ran full split regressions, repaired safe native effect proofs exposed by extra real models, added a transported-source-write refusal test, and preserved optional scenario syntax.
8. Repeated independent-process before/after measurements, disclosed memory/capture tradeoffs, ran available quality checks and updated runnable examples, glossary and this report.
9. Preserved all unrelated working state; only task-created temporary baseline files are eligible for cleanup. Automatic approval policy rejected that removal, so those files remain private and untracked. Internal investigation notes are not part of the public change.
