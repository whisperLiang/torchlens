"""F1 microbenchmark: facet indirection cost on the capture hot path.

The converged backend plan (fable_r2 D1 / opus_r2 D1) gates the Phase-4
OpCore+FacetSet decomposition on a measurement: if facet indirection costs
more than 1% of per-op capture time, the schema falls back to a flatter core
(~25 fields) plus one detail struct. This bench measures the three candidate
costs in isolation:

1. record construction: one flat ~56-field frozen slotted record (today's
   ``OpEvent`` shape) vs a ~12-field ``OpCore`` plus a slotted ``FacetSet``
   holding the same nested typed refs;
2. hot attribute reads through the extra indirection hop
   (``record.facets.function.func_name`` vs ``event.function.func_name``);
3. absent-facet checks (``facets.autograd is None``).

The verdict compares the worst per-op delta against 1% of the measured
per-op capture time in ``benchmarks/perf/s2_capture_spine_overhead.json``
(row ``trace_exhaustive``). A typical capture reads each hot field a small
constant number of times; the bench charges an aggressive 100 facet-hop
reads plus one construction per op so the verdict is conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
import json
from pathlib import Path
import statistics
import timeit
from typing import Any


def _flat_record_type() -> type:
    """Build a frozen slotted stand-in with today's OpEvent field count."""

    field_names = [f"field_{index}" for index in range(50)]
    return make_dataclass(
        "FlatOpRecord",
        [(name, object, field(default=None)) for name in field_names]
        + [("function", object, field(default=None)), ("output", object, field(default=None))],
        frozen=True,
        slots=True,
    )


@dataclass(frozen=True, slots=True)
class _FunctionFacet:
    """Function-call facet stand-in (subset of FunctionCallRef)."""

    func_name: str
    func_qualname: str
    num_args_total: int
    is_inplace: bool


@dataclass(frozen=True, slots=True)
class _OutputFacet:
    """Output facet stand-in."""

    shape: tuple[int, ...]
    dtype: str
    memory: int


@dataclass(frozen=True, slots=True)
class _FacetSet:
    """Typed optional facets; absent facet != empty facet."""

    function: _FunctionFacet | None = None
    output: _OutputFacet | None = None
    graph: object | None = None
    modules: object | None = None
    autograd: object | None = None
    transform: object | None = None
    ancestry: object | None = None
    control: object | None = None
    params: object | None = None
    annotations: object | None = None
    container: object | None = None
    policy: object | None = None
    templates: object | None = None


@dataclass(frozen=True, slots=True)
class _OpCore:
    """Required operation core (~12 fields)."""

    seq: int
    kind: str
    label_raw: str
    layer_label_raw: str
    layer_type: str
    raw_index: int
    type_index: int
    step_index: int
    pass_index: int
    parents: tuple[str, ...]
    is_bottom_level: bool
    facets: _FacetSet


def _time_us(stmt: Any, number: int, repeats: int = 7) -> float:
    """Return the median per-call microseconds for a callable."""

    timer = timeit.Timer(stmt)
    runs = [run / number * 1e6 for run in timer.repeat(repeat=repeats, number=number)]
    return statistics.median(runs)


def main() -> None:
    """Run the F1 microbench and write the JSON verdict."""

    flat_type = _flat_record_type()
    function_facet = _FunctionFacet("relu", "torch.relu", 1, False)
    output_facet = _OutputFacet((2, 16), "float32", 128)

    def build_flat() -> Any:
        return flat_type(
            field_0=1,
            field_1="op",
            field_2="relu_1_1_raw",
            field_3="relu_1_1_raw",
            field_4="relu",
            field_5=1,
            field_6=1,
            field_7=1,
            field_8=1,
            field_9=("add_1_1_raw",),
            field_10=True,
            function=function_facet,
            output=output_facet,
        )

    def build_core_facets() -> Any:
        return _OpCore(
            seq=1,
            kind="op",
            label_raw="relu_1_1_raw",
            layer_label_raw="relu_1_1_raw",
            layer_type="relu",
            raw_index=1,
            type_index=1,
            step_index=1,
            pass_index=1,
            parents=("add_1_1_raw",),
            is_bottom_level=True,
            facets=_FacetSet(function=function_facet, output=output_facet),
        )

    flat_event = build_flat()
    core_record = build_core_facets()

    def read_flat() -> Any:
        return flat_event.function.func_name  # type: ignore[attr-defined]

    def read_facets() -> Any:
        return core_record.facets.function.func_name  # type: ignore[union-attr]

    def check_absent() -> bool:
        return core_record.facets.autograd is None

    number = 200_000
    construct_flat_us = _time_us(build_flat, number)
    construct_facets_us = _time_us(build_core_facets, number)
    read_flat_us = _time_us(read_flat, number * 5)
    read_facets_us = _time_us(read_facets, number * 5)
    absent_check_us = _time_us(check_absent, number * 5)

    # Aggressive per-op charge: one construction + 100 hot facet-hop reads.
    reads_per_op = 100
    per_op_delta_us = (construct_facets_us - construct_flat_us) + reads_per_op * (
        read_facets_us - read_flat_us
    )

    baseline_path = Path(__file__).parent / "perf" / "s2_capture_spine_overhead.json"
    baseline = json.loads(baseline_path.read_text())
    exhaustive_row = next(row for row in baseline["rows"] if row["name"] == "trace_exhaustive")
    per_op_capture_us = float(exhaustive_row["per_op_us"])
    budget_us = per_op_capture_us * 0.01
    verdict = "DECOMPOSE" if per_op_delta_us <= budget_us else "FLAT_CORE_FALLBACK"

    result = {
        "schema": "torchlens.f1_facet_indirection.v1",
        "construct_flat_us": construct_flat_us,
        "construct_core_facets_us": construct_facets_us,
        "read_flat_us": read_flat_us,
        "read_facet_hop_us": read_facets_us,
        "absent_facet_check_us": absent_check_us,
        "reads_per_op_charged": reads_per_op,
        "per_op_delta_us": per_op_delta_us,
        "per_op_capture_us_baseline": per_op_capture_us,
        "budget_us_1pct": budget_us,
        "verdict": verdict,
    }
    out_path = Path(__file__).parent / "perf" / "f1_facet_indirection.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
