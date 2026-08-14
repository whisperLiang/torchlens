"""Backend-neutral helpers shared by the preview backends' validation modules.

The b5-parity R46-1 dedup wave hoisted the capture-side helper families into
``backends/_finalize.py`` but deferred the ``*/validation.py`` halves to the
b4-V territory owner (the copies were byte-identical across tf/mlx/paddle).
This module is that landing: validation-side helpers only, no backend
imports, safe to import from any preview backend.

The deliberately-divergent helpers stay per-backend: each backend's
``_payloads_close`` carries its own dtype tolerance bands (0e210ea2 gave mlx
fp16 its own ~1-ULP band while tf/paddle keep the shared NaN doctrine), and
unifying them would be a silent tolerance change -- exactly what the LOCKED
validation doctrine forbids doing casually.
"""

from __future__ import annotations

from typing import Any


def ops_by_label(trace: Any) -> dict[str, Any]:
    """Return materialized trace operations keyed by all known labels.

    Parameters
    ----------
    trace:
        Materialized TorchLens trace.

    Returns
    -------
    dict[str, Any]
        Operations keyed by raw, layer, and pass labels.

    Notes
    -----
    Key precedence is load-bearing under recurrence grouping: a group
    leader's RAW label doubles as the shared ``layer_label`` of every later
    pass, so naive last-writer insertion would silently rebind a capture
    record's raw label to the wrong pass. Raw labels are the immutable
    capture identity and always win; pass labels are unique; the ambiguous
    layer label resolves to its first pass.
    """

    result: dict[str, Any] = {}
    for op in getattr(trace, "layer_list", ()):
        layer_label = getattr(op, "layer_label", None)
        if isinstance(layer_label, str) and layer_label not in result:
            result[layer_label] = op
    for op in getattr(trace, "layer_list", ()):
        label = getattr(op, "label", None)
        if isinstance(label, str):
            result[label] = op
    for op in getattr(trace, "layer_list", ()):
        label_raw = getattr(op, "_label_raw", None)
        if isinstance(label_raw, str):
            result[label_raw] = op
    return result
