"""Tests for backend-neutral loop grouping adapter."""

import cProfile
import pstats
from typing import Any

import pytest
import torch

import example_models
from torchlens import trace as trace_fn
from torchlens.postprocess.loop_grouping_adapter import (
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)


def _raw_label(trace: Any, final_label: str) -> str:
    """Return the raw label corresponding to a final trace label.

    Parameters
    ----------
    trace:
        TorchLens trace with raw/final label maps.
    final_label:
        Final layer or op label.

    Returns
    -------
    str
        Raw label for ``final_label``.
    """
    return trace._final_to_raw_layer_labels[final_label]


def _raw_recurrent_member_sets(trace: Any) -> set[frozenset[str]]:
    """Collect expected recurrent member sets from a torch recurrent fixture.

    Parameters
    ----------
    trace:
        TorchLens trace for a recurrent torch model.

    Returns
    -------
    set[frozenset[str]]
        Raw-label recurrent groups with at least two members.
    """
    member_sets: set[frozenset[str]] = set()
    for op in trace.ops:
        if len(op.recurrent_ops) <= 1:
            continue
        raw_members = frozenset(_raw_label(trace, label) for label in op.recurrent_ops)
        member_sets.add(raw_members)
    return member_sets


def _neutral_graph_from_torch_recurrent_fixture(trace: Any) -> RecurrenceGroupingGraph:
    """Build a label-consistent neutral graph from a finalized torch trace.

    Parameters
    ----------
    trace:
        TorchLens trace for a recurrent torch model.

    Returns
    -------
    RecurrenceGroupingGraph
        Neutral graph using raw labels consistently for nodes and data edges.
    """
    nodes: dict[str, RecurrenceNode] = {}
    raw_labels: list[str] = []
    raw_label_set = {op._label_raw for op in trace.ops}
    # ``equivalent_ops`` holds finalized OP labels (pass-qualified), which are keyed
    # differently from the final LAYER labels in ``_final_to_raw_layer_labels`` --
    # map them back to raw labels through the ops themselves.
    op_label_to_raw = {op.label: op._label_raw for op in trace.ops}

    for op in trace.ops:
        raw_label = op._label_raw
        raw_labels.append(raw_label)
        nodes[raw_label] = RecurrenceNode(
            label=raw_label,
            raw_order=op.raw_index,
            equivalence_key=op.equivalence_class,
            equivalent_labels=tuple(
                op_label_to_raw[equiv_label]
                for equiv_label in op.equivalent_ops
                if equiv_label in op_label_to_raw
            ),
            data_parents=tuple(
                _raw_label(trace, parent)
                for parent in op.parents
                if parent in trace.layer_dict_all_keys
            ),
            data_children=tuple(
                _raw_label(trace, child)
                for child in op.children
                if child in trace.layer_dict_all_keys
            ),
            layer_label=raw_label,
            recurrent_labels=(),
            uses_params=bool(op.uses_params),
            func_name=op.func_name,
            param_barcodes=tuple(op._param_barcodes),
            retain=raw_label in raw_label_set,
            pruned=False,
        )

    return RecurrenceGroupingGraph(
        nodes=nodes,
        raw_labels=tuple(raw_labels),
        source_labels=tuple(_raw_label(trace, label) for label in trace.input_layers),
        eligible_labels=tuple(raw_labels),
    )


def test_neutral_loop_grouping_matches_torch_recurrent_fixture() -> None:
    """Neutral grouping service reproduces torch recurrent member sets."""
    torch.manual_seed(0)
    traced = trace_fn(example_models.RecurrentParamsSimple(), torch.rand(5, 5))

    graph = _neutral_graph_from_torch_recurrent_fixture(traced)
    assignments = group_recurrent_nodes(graph)
    actual_groups = {
        frozenset(assignment.recurrent_labels)
        for assignment in assignments.values()
        if assignment.num_passes > 1
    }

    assert actual_groups == _raw_recurrent_member_sets(traced)
    assert all("control" not in node.data_parents for node in graph.nodes.values())


class _ChainedTiedRecurrentNet(torch.nn.Module):
    """Tied linear + tanh applied ``num_steps`` times: one big chained loop."""

    def __init__(self, num_steps: int, dim: int = 16) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.tied = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for _ in range(self.num_steps):
            hidden = torch.tanh(self.tied(hidden))
        return hidden


class _ParallelStreamsNet(torch.nn.Module):
    """Disjoint parallel repeats: multi-root groups that must stay split.

    Two disconnected param-free ``relu(x) + 1`` chains form one candidate
    class that legitimately partitions into TWO recurrent layers, and a
    shared linear applied to three disjoint slices feeds three distinct
    single-pass heads. Both are adversarial for any pairwise-sweep early
    exit: the sweep must keep running while multiple roots remain.
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stream_a = x[:, 0]
        stream_b = x[:, 1]
        for _ in range(5):
            stream_a = torch.relu(stream_a) + 1
        for _ in range(5):
            stream_b = torch.relu(stream_b) + 1
        head_relu = torch.relu(self.shared(x[:, 0]))
        head_sigmoid = torch.sigmoid(self.shared(x[:, 1]))
        head_tanh = torch.tanh(self.shared(x[:, 2]))
        return (
            stream_a.sum() + stream_b.sum() + head_relu.sum() + head_sigmoid.sum() + head_tanh.sum()
        )


def _trace_with_adapter_find_calls(model: torch.nn.Module, x: torch.Tensor) -> tuple[Any, int]:
    """Trace ``model`` and count union-find ``find`` calls in the adapter."""
    profiler = cProfile.Profile()
    profiler.enable()
    traced = trace_fn(model, x)
    profiler.disable()
    find_calls = sum(
        call_count
        for (file_name, _line, func_name), (call_count, *_rest) in pstats.Stats(
            profiler
        ).stats.items()
        if func_name == "find" and file_name.endswith("loop_grouping_adapter.py")
    )
    return traced, find_calls


@pytest.mark.smoke
def test_pairwise_sweep_early_exit_keeps_chained_grouping_subquadratic() -> None:
    """Chained tied loop groups identically while the pair sweep stays subquadratic.

    Grouping both iso groups (tied linear and tanh, 64 members each) must not
    pay the historical O(members^2) already-unified pair tail: the quadratic
    sweep cost ~23k adapter ``find`` calls at 64 steps, the early-exit version
    ~3k. The bound is a call count, not a timing, so it is load-robust.
    """
    torch.manual_seed(0)
    num_steps = 64
    traced, find_calls = _trace_with_adapter_find_calls(
        _ChainedTiedRecurrentNet(num_steps), torch.rand(2, 16)
    )

    recurrent_passes = {op.layer_label: op.num_passes for op in traced.ops if op.num_passes > 1}
    assert len(recurrent_passes) == 2
    assert set(recurrent_passes.values()) == {num_steps}
    assert find_calls < 12_000


@pytest.mark.smoke
def test_pairwise_sweep_early_exit_preserves_multi_root_group_membership() -> None:
    """Multi-root candidate groups keep exact historical membership.

    A premature pairwise-sweep exit would either merge the two disconnected
    param-free streams into one layer or fail to accumulate each stream's
    five passes; the shared-weight heads must stay three distinct
    single-pass layers fed by one three-pass linear layer.
    """
    torch.manual_seed(0)
    traced, _ = _trace_with_adapter_find_calls(_ParallelStreamsNet(), torch.rand(2, 3, 8))

    num_passes = {op.layer_label: op.num_passes for op in traced.ops}
    relu_stream_layers = sorted(
        label for label, passes in num_passes.items() if label.startswith("relu") and passes == 5
    )
    assert len(relu_stream_layers) == 2
    add_stream_layers = sorted(
        label for label, passes in num_passes.items() if label.startswith("add") and passes == 5
    )
    assert len(add_stream_layers) == 2
    shared_linear_passes = {
        passes for label, passes in num_passes.items() if label.startswith("linear")
    }
    assert shared_linear_passes == {3}
    for head in ("sigmoid", "tanh"):
        head_passes = [p for label, p in num_passes.items() if label.startswith(head)]
        assert head_passes == [1]
