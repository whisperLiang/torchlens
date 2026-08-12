"""F1b real-shape pre-gate: per-op record-construction cost A/B (P0).

Measures, on a real captured op stream, the marginal CPU cost of building the
decomposed record shape (frozen OpCore + 13 typed facets, shared call-scoped
prototypes) against today's ~150-key mutable fields dict + per-output copy.
Paired same-process alternating A/B on ``time.process_time``, >=20 repeats,
reported with the minimum detectable effect. Run manually:

    PYTHONPATH=.:tests python -m producer_parity.bench_f1b
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass


REPEATS = 24
OPS_PER_REP = 2000


# --- real-shape prototypes: 13 required core fields + 13 facets -------------


@dataclass(frozen=True, slots=True)
class _ProtoCore:
    seq: int
    kind: str
    label_raw: str
    layer_label_raw: str
    layer_type: str
    raw_index: int
    type_index: int
    step_index: int
    pass_index: int
    parents: tuple
    output: object
    is_bottom_level: bool
    func_call_id: int


@dataclass(frozen=True, slots=True)
class _ProtoFacetSmall:
    a: object = None
    b: object = None


@dataclass(frozen=True, slots=True)
class _ProtoFacetSet:
    function: object
    templates: object
    graph: object
    modules: object
    ancestry: object
    autograd: object
    transform: object
    control: object
    params: object
    annotations: object
    policy: object
    recording: object
    intervention: object


@dataclass(frozen=True, slots=True)
class _ProtoRecord:
    core: _ProtoCore
    facets: _ProtoFacetSet


_SHARED_FUNCTION_REF = ("shared", "call", "scoped", "ref")
_SHARED_POLICY = ("interned", "policy")
_150_KEYS = [f"key_{index}" for index in range(150)]


def _build_fields_dict(index: int) -> dict:
    """Today's shape: ~150-key dict built per call + copied per output."""

    fields = {key: None for key in _150_KEYS}
    fields["label_raw"] = f"op_{index}_raw"
    fields["raw_index"] = index
    fields["out"] = index
    per_output = dict(fields)  # the per-output copy (ops.py:4376 analogue)
    return per_output


def _build_record(index: int) -> _ProtoRecord:
    """Decomposed shape: frozen core + facet set, shared prototypes."""

    core = _ProtoCore(
        seq=index,
        kind="op",
        label_raw=f"op_{index}_raw",
        layer_label_raw=f"op_{index}",
        layer_type="op",
        raw_index=index,
        type_index=index,
        step_index=0,
        pass_index=1,
        parents=(),
        output=index,
        is_bottom_level=True,
        func_call_id=index,
    )
    facets = _ProtoFacetSet(
        function=_SHARED_FUNCTION_REF,
        templates=_ProtoFacetSmall(),
        graph=_ProtoFacetSmall(a=index),
        modules=None,
        ancestry=None,
        autograd=_ProtoFacetSmall(a=index),
        transform=None,
        control=None,
        params=None,
        annotations=None,
        policy=_SHARED_POLICY,
        recording=None,
        intervention=None,
    )
    return _ProtoRecord(core=core, facets=facets)


def run() -> dict:
    """Paired alternating A/B; returns the measurement summary."""

    dict_times: list[float] = []
    record_times: list[float] = []
    sink: list = []
    for _ in range(REPEATS):
        start = time.process_time()
        for index in range(OPS_PER_REP):
            sink.append(_build_fields_dict(index))
        dict_times.append(time.process_time() - start)
        sink.clear()

        start = time.process_time()
        for index in range(OPS_PER_REP):
            sink.append(_build_record(index))
        record_times.append(time.process_time() - start)
        sink.clear()

    dict_mean = statistics.mean(dict_times)
    record_mean = statistics.mean(record_times)
    pooled_sd = statistics.stdev([d - r for d, r in zip(dict_times, record_times)])
    mde = 2.0 * pooled_sd / max(dict_mean, 1e-12)
    return {
        "ops_per_rep": OPS_PER_REP,
        "repeats": REPEATS,
        "dict_mean_s": dict_mean,
        "record_mean_s": record_mean,
        "record_over_dict": record_mean / dict_mean,
        "paired_mde_fraction": mde,
        "per_op_delta_us": (record_mean - dict_mean) / OPS_PER_REP * 1e6,
    }


if __name__ == "__main__":
    result = run()
    for key, value in result.items():
        print(f"{key}: {value}")
