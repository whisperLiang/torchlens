"""ReplayProgram and split capability-report tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from v2_helpers import (
    plan_fixture,
    split_request,
)

from torchlens.split.adapters.torch import TorchSplitAdapter
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.graph import SplitTraceGraph, SplitTraceNode
from torchlens.split.program import (
    build_capability_report,
    ensure_capability_report_supported,
    lower_replay_program,
)
from torchlens.utils.display import identity


def _node(
    label: str,
    *,
    parents: tuple[str, ...] = (),
    target: Any | None = None,
    is_input: bool = False,
    is_output: bool = False,
    raw_index: int = 0,
) -> SplitTraceNode:
    """Create a minimal split graph node for ReplayProgram tests."""

    return SplitTraceNode(
        label=label,
        raw_label=f"raw_{label}",
        canonical_id=label,
        backend="torch",
        raw_index=raw_index,
        op_type=label,
        target=target,
        func_call_id=None,
        args_template=None,
        kwargs_template=None,
        parents=parents,
        children=(),
        output_ref=None,
        module_path=None,
        output_shape=(2, 3),
        symbolic_output_shape=None,
        dtype="torch.float32",
        requires_grad=False,
        output_container_path=(),
        output_container_spec=None,
        is_input=is_input,
        is_output=is_output,
        is_buffer=False,
        is_buffer_only_source=False,
        is_param_source=False,
        param_refs=(),
        replay_source_policy="constant",
        op=SimpleNamespace(out=None),
    )


def test_replay_program_preflight_rejects_targetless_suffix_compute() -> None:
    """ReplayProgram preflight fails closed before targetless compute nodes replay."""

    graph = SplitTraceGraph(
        backend="torch",
        nodes=(
            _node("input", is_input=True, raw_index=0),
            _node("h", parents=("input",), target=identity, raw_index=1),
            _node("unsupported", parents=("h",), raw_index=2),
            _node("output", parents=("unsupported",), is_output=True, raw_index=3),
        ),
        input_node_ids=("input",),
        output_node_ids=("output",),
        graph_shape_hash="abc",
        traced_batch_size=2,
    )
    spec = split_request("through:h")
    plan = plan_fixture(graph, spec)
    prefix_program = lower_replay_program(graph, plan, spec, segment="prefix")
    suffix_program = lower_replay_program(graph, plan, spec, segment="suffix")

    report = build_capability_report(
        TorchSplitAdapter(),
        graph,
        plan,
        spec,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
    )

    assert report.preflight_ok is False
    assert any("missing replay target" in reason for reason in report.preflight.details)
    with pytest.raises(SplitUnsupportedError, match="preflight"):
        ensure_capability_report_supported(report, spec)
