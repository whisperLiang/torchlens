# Split runtime error analysis — 2026-11-02

These observations use committed source `1026a6650bd010cea64266e50348efcfa667137d`
on one NVIDIA Orin with PyTorch 2.11.0 and CUDA 12.6. They extend the
[acceptance report](split_runtime_acceptance.md) with separately reproduced
causes. A failure in one category does not establish the cause of another.

| Observation | Supported cause | Does float64 resolve it? |
| --- | --- | --- |
| RF-DETR train B=2 probe, large logits mismatch | Captured `split(batch_size)` operand is not rewritten | No; reproduced in the real model and a deterministic CPU control |
| ResNet/YOLO heterogeneous train buffer mismatch | Updates to transported temporary buffers do not persist in effective state | No; reproduced in a float64 Conv/BatchNorm control |
| DistilBERT heterogeneous train, broad output mismatch | CPU/CUDA Dropout generators assign different masks despite the same seed | No; native float32 and float64 controls retain mask differences |
| Small float32 gradient/update mismatches | Exact numerical mechanism not yet established per failing call | Separate precision controls are evidence, not a changed float32 verdict |

## RF-DETR: dynamic split operand

RF-DETR 1.8.3 training groups 13 sets of queries along the batch axis for
self-attention, then splits that axis using the original batch size and joins
the results along the query axis. This inverse grouping must change its split
size when the runtime batch changes.

In the first decoder layer, the first structural divergence is
`split_40_1038:1`:

| Execution | Split size / axis | Output shape |
| --- | --- | --- |
| Captured B=1 | 1 / 0 | `[1, 300, 256]` |
| Native B=2 witness | 2 / 0 | `[2, 300, 256]` |
| B=2 replay | 1 / 0 | `[1, 300, 256]` |

The replay uses the captured scalar rather than the B=2 scalar. It creates
extra chunks, while later replay recipes consume only the output occurrences
recorded at capture. Broadcasting can subsequently restore the final output
shape while mixing the wrong values. Matching the final tensor shape therefore
does not establish semantic equivalence.

The detailed real-model diagnostic measured maximum absolute logits differences
of 6.046344757 in float32 and 4.577649390 in float64. Both first diverged at the
same split and shapes. All 21 observed Dropout calls had probability zero.
This failure is independent of active Dropout and low floating-point precision.

The source distinction is in `shape_program.py`: split output shapes participate
in shape inference, but `_uses_shape_descriptor()` / `_compile_recipes()` do not
lower a dynamic split-size operand. Knowing the intended output shape does not
rewrite that operand. The existing B=2 numerical probe correctly refuses dynamic
batch authorization rather than exposing the incorrect replay as verified.

A small deterministic CPU model reproduces the same failure in float32 and
float64, without attention, Dropout or a detector checkpoint:

```python
class Grouped(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        grouped = torch.cat(x.split(2, dim=1), dim=0) * 2
        return x + torch.cat(grouped.split(batch, dim=0), dim=1)
```

For input `[2, 6, 8]`, grouping produces `[6, 2, 8]`. The captured split size 1
produces six chunks instead of the required three chunks of size 2. Consuming
three recorded chunks produces `[1, 6, 8]`, which the final addition broadcasts.
The native result is `3 * x`; the incorrect replay mixes samples and query groups.

Intermediate diagnostics retain full B=2 outputs to locate the first divergence;
ordinary acceptance retains its lean witness. Both paths exhibit the failure.
Diagnostic observations do not replace acceptance, modify its comparison result,
or change tolerances.

## Heterogeneous BatchNorm: state storage

A Conv2d/BatchNorm2d/Tanh control in float64 reproduces state loss in both
CPU→CUDA and CUDA→CPU placements. A legal prefix includes the convolution and
counter write. The current planner accepts this partition with its normal
effect/alias checks.

The native and split losses agree, and both batch counters advance from zero
to one. Native running statistics change; effective split bindings retain zero
running means and unit running variances. For one CPU→CUDA run, the native mean
became `[0.0026199840, 0.0057245835, -0.0113552734, 0.0069862028]`, while both
published split bindings remained zero.

The suffix-training helper preserves writable buffer storage before transport.
`SplitRuntime._run_suffix_unchecked()` subsequently transports the boundary
through `boundary.to()`. Across devices that creates temporary tensor storage;
native BatchNorm updates the temporary values, which are not committed back to
the effective state owners/replicas. Later state inspection and recuts therefore
read stale statistics. Same-device storage preservation alone does not cover
this transport case.

The real heterogeneous ResNet failures occur in `buffer_error()`, with 511 of
512 values differing and maximum absolute errors around 0.11–0.13. These state
failures are distinct from a small gradient tolerance exceedance. A separate
all-suffix mutation candidate was refused by the planner; no refusal was bypassed
to obtain the legal-prefix reproduction.

## Dropout: device-specific random mapping

With probability 0.1, seed 317 and 16,384 all-one elements, native CPU/CUDA
Dropout masks disagree at 2,913 positions in float32 and 2,908 in float64.
Reseeding and repeating on the same device gives identical masks. The different
devices do not assign the same random draws to the same logical elements merely
because their seeds match.

DistilBERT's actual p=0.1 single-GPU training passed both sequential and DAG
validation, including all 12 optimizer/batch updates per kind. Its heterogeneous
training output differences are consistent with this independently demonstrated
generator mapping difference. The native control supports that mechanism without
substituting for the model-level comparisons.

## Remaining numerical differences and repair constraints

Some failures are much smaller, such as six of 49,152 ResNet single-GPU input
gradient values exceeding tolerance with maximum absolute error 1.256e-4.
Operation ordering, kernel choices, layout and rounding are possible contributors;
the failing calls have not established one exact mechanism. Higher precision
controls must stay separately labelled, and do not turn float32 failures into
passes.

Repairs must retain the existing protections: rewrite only split operands with
supported provenance, give mutable transported buffers explicit persistent state
semantics, and disclose the cross-device RNG equivalence policy. Replacing every
integer equal to batch size, ignoring buffer comparisons, disabling Dropout,
switching to eval, or widening tolerances would not repair the reproduced causes.
