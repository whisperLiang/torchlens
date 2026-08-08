"""Tests for backend-neutral loop grouping adapter."""

import cProfile
import itertools as it
import pstats
from collections import Counter, defaultdict
from typing import Any

import pytest
import torch

import example_models
import torchlens.postprocess.loop_grouping_adapter as lga
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


class _ExplicitCatCellRNN(torch.nn.Module):
    """Explicit ``tanh(cell(cat))`` unrolled loop: the degenerate entry regime."""

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.cell = torch.nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            hidden = torch.tanh(self.cell(torch.cat([x[:, step], hidden], dim=1)))
        return hidden


class _AddCellRNN(torch.nn.Module):
    """Explicit ``tanh(w(h) + x_t)`` unrolled loop: the add-cell battery regime."""

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.w = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            hidden = torch.tanh(self.w(hidden) + x[:, step])
        return hidden


class _HandLSTM(torch.nn.Module):
    """Hand-rolled LSTM cell loop: entry admissions genuinely fire here."""

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.wi = torch.nn.Linear(2 * dim, dim)
        self.wf = torch.nn.Linear(2 * dim, dim)
        self.wo = torch.nn.Linear(2 * dim, dim)
        self.wg = torch.nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], x.shape[2])
        cell = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            joint = torch.cat([x[:, step], hidden], dim=1)
            gate_i = torch.sigmoid(self.wi(joint))
            gate_f = torch.sigmoid(self.wf(joint))
            gate_o = torch.sigmoid(self.wo(joint))
            gate_g = torch.tanh(self.wg(joint))
            cell = gate_f * cell + gate_i * gate_g
            hidden = gate_o * torch.tanh(cell)
        return hidden


def _unfiltered_pf_partition_class_oracle(
    workspace: Any,
    members: list,
    signatures: dict,
    parent_colors: dict,
    consumer_site_frames: dict,
    complete_consumer_sites: set,
    class_of: dict,
    realizations: dict,
    reach_memo: Any,
) -> list:
    """Verbatim pre-prefilter ``_pf_partition_class``: the byte-identity oracle.

    This is the unfiltered O(members^2) entry sweep exactly as shipped before
    the entry-admission prefilter, kept as a reference implementation so the
    prefiltered production sweep can be asserted union-identical on every real
    partition call issued while tracing recurrent fixtures.
    """
    parent_map = {member: member for member in members}

    def find(label: str) -> str:
        while parent_map[label] != label:
            parent_map[label] = parent_map[parent_map[label]]
            label = parent_map[label]
        return label

    def union(label1: str, label2: str) -> None:
        root1, root2 = find(label1), find(label2)
        if root1 != root2:
            parent_map[root2] = root1

    cohorts: dict = defaultdict(list)
    for member in members:
        cohorts[frozenset(signatures[member].items())].append(member)
    cohort_sizes = {signature: len(cohort) for signature, cohort in cohorts.items()}

    for signature, cohort in cohorts.items():
        if len(cohort) < 2:
            continue
        invariant_fed = all(color[0] == "ext" for color, _ in signature)
        distinct_roots = len({find(member) for member in cohort})
        pair_iter = it.chain(zip(cohort, cohort[1:]), it.combinations(cohort, 2))
        for member1, member2 in pair_iter:
            if find(member1) == find(member2):
                continue
            consumers1 = consumer_site_frames[member1]
            consumers2 = consumer_site_frames[member2]
            if (
                sum(count for _, count in signature) >= 2
                and consumers1
                and consumers2
                and set(consumers1) <= complete_consumer_sites
                and set(consumers2) <= complete_consumer_sites
                and consumers1.keys() != consumers2.keys()
            ):
                continue
            if lga._reaches_forward(workspace, member1, member2, reach_memo):
                union(member1, member2)
                distinct_roots -= 1
                if distinct_roots == 1:
                    break
            elif invariant_fed and lga._pf_child_route_allows(
                workspace, member1, member2, class_of, reach_memo
            ):
                union(member1, member2)
                distinct_roots -= 1
                if distinct_roots == 1:
                    break

    equal_component_sizes = Counter(find(member) for member in members)
    realized_equal_members = {
        member for member in members if equal_component_sizes[find(member)] > 1
    }
    for index, entry in enumerate(members):
        if entry in realized_equal_members:
            continue
        entry_signature = frozenset(signatures[entry].items())
        for target in members[index + 1 :]:
            if frozenset(signatures[target].items()) == entry_signature:
                continue
            if find(entry) == find(target):
                break
            if lga._pf_entry_union_allowed(
                workspace,
                entry,
                target,
                signatures,
                parent_colors,
                consumer_site_frames,
                complete_consumer_sites,
                cohort_sizes,
                realizations,
                reach_memo,
            ):
                union(entry, target)
                break

    groups: dict = defaultdict(list)
    for member in members:
        groups[find(member)].append(member)
    return [
        sorted(group, key=lambda label: workspace.nodes[label].raw_order)
        for group in groups.values()
    ]


@pytest.mark.smoke
def test_entry_sweep_prefilter_matches_unfiltered_oracle(monkeypatch: Any) -> None:
    """Prefiltered entry sweep is partition-identical to the unfiltered oracle.

    Every real ``_pf_partition_class`` call issued while tracing the battery
    (degenerate cat-cell, add-cell, admissions-firing hand LSTM, and the
    multi-root parallel-streams adversary) must return exactly the groups the
    verbatim pre-prefilter sweep returns, in the same order.
    """
    production = lga._pf_partition_class
    compared_calls = {"count": 0}

    def comparing_partition(*args: Any, **kwargs: Any) -> list:
        produced = production(*args, **kwargs)
        oracle = _unfiltered_pf_partition_class_oracle(*args, **kwargs)
        assert produced == oracle
        compared_calls["count"] += 1
        return produced

    monkeypatch.setattr(lga, "_pf_partition_class", comparing_partition)
    torch.manual_seed(0)
    trace_fn(_ExplicitCatCellRNN(32), torch.rand(2, 32, 8))
    trace_fn(_AddCellRNN(32), torch.rand(2, 32, 8))
    trace_fn(_HandLSTM(24), torch.rand(2, 24, 8))
    trace_fn(_ParallelStreamsNet(), torch.rand(2, 3, 8))
    assert compared_calls["count"] > 0


@pytest.mark.smoke
def test_entry_sweep_prefilter_empties_degenerate_pair_triangle(monkeypatch: Any) -> None:
    """The degenerate long-loop regime issues ZERO entry-admission pair calls.

    Every cat-cell signature is a distinct singleton cohort of bare colors, so
    all three admission arms are structurally unreachable: the prefilter must
    empty the pair triangle outright (the pre-prefilter sweep paid C(N-1, 2)
    calls here) while grouping still assigns every loop op its full pass
    count. The hand LSTM guards the other direction: admissions still fire
    through the prefilter.
    """
    real_allowed = lga._pf_entry_union_allowed
    outcomes: list[bool] = []

    def spying_allowed(*args: Any, **kwargs: Any) -> bool:
        allowed = real_allowed(*args, **kwargs)
        outcomes.append(allowed)
        return allowed

    monkeypatch.setattr(lga, "_pf_entry_union_allowed", spying_allowed)
    torch.manual_seed(0)
    num_steps = 32
    traced = trace_fn(_ExplicitCatCellRNN(num_steps), torch.rand(2, num_steps, 8))
    assert outcomes == []
    recurrent_passes = {op.layer_label: op.num_passes for op in traced.ops if op.num_passes > 1}
    assert set(recurrent_passes.values()) == {num_steps}

    outcomes.clear()
    trace_fn(_HandLSTM(24), torch.rand(2, 24, 8))
    assert any(outcomes)
