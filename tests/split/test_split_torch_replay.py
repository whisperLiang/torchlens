"""Torch generated-eager split replay tests."""

from __future__ import annotations

from v2_helpers import split_request

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.split.adapters.torch import GeneratedSuffix, _is_template_dict
from torchlens.split.boundary import ReplayBoundary
from torchlens.split.errors import SplitUnsupportedError
from torchlens.split.graph import SplitTraceGraph, SplitTraceNode
from torchlens.split.planner import SplitPlan
from torchlens.split.shape import SymbolicShape
from torchlens.split import BoundarySchema


def _assert_close(left: Any, right: Any) -> None:
    """Recursively assert tensor structures are close."""

    if isinstance(left, torch.Tensor):
        assert torch.allclose(left, right, atol=1e-5, rtol=1e-4)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_close(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for l_item, r_item in zip(left, right):
            _assert_close(l_item, r_item)
    else:
        assert left == right


class SharedLabelMultiOutput(nn.Module):
    """Use every output of one call whose leaves share a display label."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second, third = torch.chunk(x, 3, dim=1)
        return first + second * 2 + third * 3


class TinyMlp(nn.Module):
    """Linear-ReLU-linear toy model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_multi_output_boundaries_bind_canonical_values_not_display_labels() -> None:
    """Every chunk output survives before/after boundaries with duplicate labels."""

    model = SharedLabelMultiOutput().eval()
    example = torch.randn(2, 9)
    seed_runtime = tl.split.prepare(model, example, split_request("50%"))
    captured_chunk_nodes = tuple(
        node for node in seed_runtime.trace_graph.compute_nodes if node.op_type == "chunk"
    )
    assert len(captured_chunk_nodes) == 3
    chunk_ids = {node.canonical_id for node in captured_chunk_nodes}
    seed_runtime.trace_graph = replace(
        seed_runtime.trace_graph,
        nodes=tuple(
            replace(node, label="shared_chunk_output") if node.canonical_id in chunk_ids else node
            for node in seed_runtime.trace_graph.nodes
        ),
    )
    chunk_nodes = tuple(
        node for node in seed_runtime.trace_graph.compute_nodes if node.op_type == "chunk"
    )

    assert len({node.label for node in chunk_nodes}) == 1
    output_value_ids = {f"value:{node.canonical_id}" for node in chunk_nodes}
    assert any(
        set(operation.output_value_ids) == output_value_ids
        for operation in seed_runtime.graph_ir.ops
    )

    for node in chunk_nodes:
        for point in (tl.split.before(node.canonical_id), tl.split.after(node.canonical_id)):
            runtime = seed_runtime.at(point)
            assert runtime.plan.boundary_spec.keys() == runtime.plan.boundary_bindings.keys()
            _assert_close(runtime.replay(example), model(example))


class TinyCnn(nn.Module):
    """Conv-ReLU-conv toy model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class ResidualMlp(nn.Module):
    """Residual/skip frontier toy model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        y = self.relu(h)
        return self.head(y) + h


class DictOutput(nn.Module):
    """Multi-output/dict reconstruction toy model."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        values, indices = torch.max(x, dim=1)
        idx = indices.float()
        return {"values": values, "indices": idx, "sum": values + idx}


def _split_node(
    label: str,
    *,
    parents: tuple[str, ...] = (),
    target: Any | None = None,
    is_input: bool = False,
    is_output: bool = False,
    op_out: torch.Tensor | None = None,
) -> SplitTraceNode:
    """Create a minimal split node for adapter unit tests."""

    return SplitTraceNode(
        label=label,
        raw_label=f"raw_{label}",
        canonical_id=label,
        backend="torch",
        raw_index=0,
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
        symbolic_output_shape=SymbolicShape((2, 3)),
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
        op=SimpleNamespace(out=torch.zeros(2, 3) if op_out is None else op_out),
    )


def test_mlp_replay_equivalence() -> None:
    """Prefix + suffix replay matches a tiny MLP."""

    torch.manual_seed(0)
    model = TinyMlp().eval()
    x = torch.randn(2, 4)

    runtime = tl.split.prepare(model, x, split_request("50%"))

    assert runtime.validate_equivalence(model, (x,))
    _assert_close(runtime.replay(x), model(x))


def test_cnn_replay_equivalence() -> None:
    """Prefix + suffix replay matches a tiny CNN."""

    torch.manual_seed(0)
    model = TinyCnn().eval()
    x = torch.randn(2, 3, 8, 8)

    runtime = tl.split.prepare(model, x, split_request("50%"))

    assert runtime.validate_equivalence(model, (x,))


def test_before_and_after_boundaries_work() -> None:
    """Both explicit boundary directions execute."""

    torch.manual_seed(0)
    model = TinyMlp().eval()
    x = torch.randn(2, 4)

    after = tl.split.prepare(model, x, split_request("after:relu"))
    before = tl.split.prepare(model, x, split_request("before:fc2"))

    _assert_close(after.replay(x), model(x))
    _assert_close(before.replay(x), model(x))


def test_identity_output_marker_reconstructs_from_graph_parent() -> None:
    """Suffix replay reconstructs synthetic output markers without executing them."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 4), nn.Dropout(p=0.0)).eval()
    x = torch.ones(2, 6)

    runtime = tl.split.prepare(model, x, split_request("after:dropout", dynamic_batch=(1, 3)))

    for batch in (1, 3):
        replay_x = torch.ones(batch, 6)
        _assert_close(runtime.replay(replay_x), model(replay_x))


def test_residual_frontier_includes_skip_tensor() -> None:
    """Residual suffix needs both primary and skip boundary tensors."""

    torch.manual_seed(0)
    model = ResidualMlp().eval()
    x = torch.randn(2, 4)

    runtime = tl.split.prepare(model, x, split_request("after:relu"))
    boundary = runtime.run_prefix(x)

    assert len(boundary.tensors) >= 2
    assert {item.role for item in boundary.spec.values()} >= {"primary", "skip"}
    _assert_close(runtime.run_suffix(boundary), model(x))


def test_multi_output_dict_reconstruction() -> None:
    """Dict outputs are reconstructed from split replay leaves."""

    model = DictOutput().eval()
    x = torch.randn(2, 5)

    runtime = tl.split.prepare(model, x, split_request("50%"))

    _assert_close(runtime.replay(x), model(x))


def test_empty_tuple_template_is_not_dict_template() -> None:
    """Empty tuple args must round-trip as tuples, not dicts."""

    assert not _is_template_dict(())


def test_targetless_suffix_compute_node_raises() -> None:
    """Target-less compute nodes cannot replay from captured outputs."""

    boundary_spec = {
        "h": BoundarySchema(
            value_id="h",
            container_path=(),
            role="primary",
            alias_group=None,
            source_kind="boundary",
            label="h",
            backend="torch",
            module_path=None,
            op_type="relu",
            shape=SymbolicShape((2, 3)),
            dtype="torch.float32",
            requires_grad=False,
        )
    }
    graph = SplitTraceGraph(
        backend="torch",
        nodes=(
            _split_node("h"),
            _split_node("unsupported", parents=("h",), op_out=torch.ones(2, 3)),
            _split_node("output", parents=("unsupported",), is_output=True),
        ),
        input_node_ids=(),
        output_node_ids=("output",),
        graph_shape_hash="abc",
        traced_batch_size=2,
    )
    plan = SplitPlan(
        split_id="split",
        boundary_kind="after",
        target_node_id="h",
        prefix_node_ids=frozenset({"h"}),
        suffix_node_ids=frozenset({"unsupported", "output"}),
        boundary_node_ids=("h",),
        boundary_spec=boundary_spec,
    )
    suffix = GeneratedSuffix(
        graph=graph,
        plan=plan,
        spec=split_request("after:h"),
        node_ids=plan.suffix_node_ids,
        use_live_param_sources=True,
    )
    boundary = ReplayBoundary(
        backend="torch",
        tensors={"h": torch.randn(2, 3)},
        spec=boundary_spec,
        metadata={"split_id": "split", "batch_symbol": "B"},
    )

    with pytest.raises(SplitUnsupportedError, match="no callable target"):
        suffix(boundary)
