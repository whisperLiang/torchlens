"""FP64 training cuts must preserve storage effects as well as value edges."""

from __future__ import annotations

import runpy
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens.split import GraphPartition, PlacementPlan, SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError

DRIVER = Path(__file__).resolve().parents[2] / "benchmarks/split_graph_partition.py"


class NormBranches(nn.Module):
    """Interleave unchanged native BatchNorm and inplace activation branches."""

    def __init__(self, *, shared: bool = False) -> None:
        super().__init__()
        self.left = nn.Conv2d(2, 2, 1)
        self.right = nn.Conv2d(2, 2, 1)
        self.left_norm = nn.BatchNorm2d(2)
        self.right_norm = self.left_norm if shared else nn.BatchNorm2d(2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Allocate both branches before either normalization or local write."""
        left = self.left(x)
        right = self.right(x)
        left = self.activation(self.left_norm(left))
        right = self.activation(self.right_norm(right))
        return (left + right).sin()


@pytest.mark.parametrize("retain_trace", [False, True])
@pytest.mark.parametrize(
    "prefix,suffix", [("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cpu"), ("cuda", "cuda")]
)
def test_independent_norm_mutations_train_and_repartition_in_fp64(
    retain_trace: bool, prefix: str, suffix: str
) -> None:
    """A/B/C/A keeps native buffers, connected gradients and Adam history current."""
    driver = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    if "cuda" in (prefix, suffix) and not torch.cuda.is_available():
        pytest.skip("Real CUDA is required for branch buffer migration")
    model = NormBranches().to(prefix).double().train()
    reference = deepcopy(model)
    context = prepare_graph(
        model,
        torch.randn(1, 2, 3, 3, dtype=torch.float64, device=prefix),
        features=SplitFeatures(training=True, retain_trace=retain_trace),
        placement=PlacementPlan.across(prefix, suffix),
    )
    nodes = context.trace_graph.compute_nodes
    # Each is an actual inversion in captured native compute order, not a
    # sequential fallback. In A the right projection crosses the left BN.
    groups = (
        {"left", "left_norm"},
        {"right", "right_norm"},
    )
    partitions = []
    for modules in groups:
        selected = {n.canonical_id for n in nodes if n.module_path in modules}
        selected.update(
            n.canonical_id
            for n in nodes
            if n.op_type == "relu" and any(parent in selected for parent in n.parents)
        )
        partition = GraphPartition(frozenset(selected))
        assert driver["noncontiguous"](context.trace_graph, partition.prefix_node_ids)
        partitions.append(partition)
    partitions.append(
        GraphPartition(
            partitions[0].prefix_node_ids
            | frozenset(
                n.canonical_id
                for n in nodes
                if n.module_path == "right_norm" and n.op_type == "add"
            )
        )
    )
    assert driver["noncontiguous"](context.trace_graph, partitions[2].prefix_node_ids)
    runtime = context.materialize(context.analyze(partitions[0]))
    parameters = driver["effective_parameters"](runtime)
    actual_opt = driver["make_optimizer"]("adam", parameters.values())
    expected_opt = driver["make_optimizer"]("adam", reference.parameters())
    for index, batch in zip((0, 1, 2, 0), (1, 2, 4, 8), strict=True):
        runtime = runtime.repartition(partitions[index])
        current = driver["effective_parameters"](runtime)
        if prefix == suffix:
            assert all(current[key] is value for key, value in parameters.items())
        actual_opt = driver["rebind_optimizer"]("adam", actual_opt, parameters, current)
        parameters = current
        errors = driver["training_step"](
            runtime,
            reference,
            torch.randn(batch, 2, 3, 3, dtype=torch.float64, device=prefix, requires_grad=True),
            actual_opt,
            expected_opt,
        )
        assert max(v for k, v in errors.items() if k.endswith("_error")) < 1e-10
    assert reference.left_norm.num_batches_tracked.item() == 4
    assert reference.right_norm.num_batches_tracked.item() == 4
    assert model.activation.inplace


def test_shared_batchnorm_cannot_reverse_its_state_writes() -> None:
    """Data-independent calls to the same BatchNorm still have a WAW edge."""
    model = NormBranches(shared=True).double().train()
    context = prepare_graph(
        model,
        torch.randn(1, 2, 3, 3, dtype=torch.float64),
        features=SplitFeatures(training=True, retain_trace=False),
    )
    nodes = context.trace_graph.compute_nodes
    right_projection = next(n for n in nodes if n.module_path == "right")
    norms = [n for n in nodes if n.op_type == "batchnorm"]
    # The second BN's input is data independent of the first BN's output.
    selected = {right_projection.canonical_id, norms[1].canonical_id}
    with pytest.raises(SplitUnsupportedError) as caught:
        context.analyze(GraphPartition(frozenset(selected)))
    assert caught.value.context.reason in {
        "partition_effect_order_unproven",
        "partition_reverse_dependency",
    }


def test_read_before_alias_write_is_a_real_effect_dependency() -> None:
    """A pre-write reader cannot be moved after a write via a view alias."""

    class AliasOrder(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Read a fresh tensor before mutating an alias of that storage."""
            fresh = x + 1
            before = fresh.square()
            alias = fresh.view_as(x)
            written = alias.relu_()
            return before + written

    context = prepare_graph(AliasOrder(), torch.ones(2, 4, dtype=torch.float64))
    selected = frozenset(
        n.canonical_id
        for n in context.trace_graph.compute_nodes
        if n.op_type != "square" and n is not context.trace_graph.compute_nodes[-1]
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        context.analyze(GraphPartition(selected))
    assert caught.value.context.reason == "partition_effect_order_unproven"
    diagnostic = caught.value._partition_diagnostic
    assert diagnostic["category"] == "true_effect_conflict"
    assert diagnostic["effect_conflicts"]
    assert diagnostic["possible_aliases"]
    assert diagnostic["intervening_calls"]
    assert diagnostic["accesses_buffer"] is False


@pytest.mark.parametrize("when", ["analysis", "prefix", "suffix"])
def test_new_buffer_alias_invalidates_cached_effect_proof(when: str) -> None:
    """Future buffer replacements cannot reuse capture-time disjointness evidence."""
    model = NormBranches().double().train()
    context = prepare_graph(
        model,
        torch.randn(1, 2, 3, 3, dtype=torch.float64),
        features=SplitFeatures(training=True, retain_trace=False),
    )
    nodes = context.trace_graph.compute_nodes
    selected = {n.canonical_id for n in nodes if n.module_path in {"left", "left_norm"}}
    selected.update(
        n.canonical_id
        for n in nodes
        if n.op_type == "relu" and any(p in selected for p in n.parents)
    )
    partition = GraphPartition(frozenset(selected))
    runtime = context.materialize(context.analyze(partition))
    x = torch.randn(1, 2, 3, 3, dtype=torch.float64)
    boundary = runtime.run_training_prefix(x) if when == "suffix" else None
    # A distinct tensor object over the first branch's storage, not a tied
    # buffer identity that the original effect graph could already order.
    model.right_norm.running_mean = model.left_norm.running_mean.view_as(
        model.left_norm.running_mean
    )
    with pytest.raises(SplitUnsupportedError) as caught:
        if when == "analysis":
            context.analyze(partition)
        elif when == "prefix":
            runtime.run_training_prefix(x)
        else:
            runtime.run_suffix(boundary)
    assert caught.value.context.reason == "partition_effect_order_unproven"
    diagnostic = caught.value._partition_diagnostic
    assert diagnostic["category"] == "cross_segment_state_conflict"
    assert diagnostic["source_alias_conflict"][2] == "overlap"


def test_effect_candidate_unions_extend_single_ancestor_search() -> None:
    """Bounded discovery includes genuine cuts missing from single-call ancestry."""
    from torchlens.split.candidates import _effect_closed_partitions

    class ThreeBranches(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Leave multiple independent subgraphs to combine in the prefix."""
            left, middle, right = x + 1, x + 2, x + 3
            return left.sin() + middle.cos() + right.tanh()

    context = prepare_graph(ThreeBranches(), torch.ones(1, 4, dtype=torch.float64))
    driver = runpy.run_path(str(DRIVER), run_name="benchmark_test")
    ancestors = driver["candidates"](context.trace_graph, "noncontiguous_dag", strategy="ancestors")
    effects = _effect_closed_partitions(context.trace_graph)
    assert set(effects) - set(ancestors)
    assert len(_effect_closed_partitions(context.trace_graph, limit=4)) <= 4
    assert _effect_closed_partitions(context.trace_graph, limit=0) == ()
    for partition in effects:
        assert driver["noncontiguous"](context.trace_graph, partition.prefix_node_ids)
        analysis = context.analyze(partition)
        assert set(analysis.plan.boundary_node_ids) & partition.prefix_node_ids


def test_unknown_runtime_storage_relation_is_not_reported_as_proven_overlap() -> None:
    """A meta input has no addressable storage; the guard must disclose uncertainty."""
    model = NormBranches().double().train()
    context = prepare_graph(
        model,
        torch.randn(1, 2, 3, 3, dtype=torch.float64),
        features=SplitFeatures(training=True, retain_trace=False),
    )
    selected = {
        n.canonical_id
        for n in context.trace_graph.compute_nodes
        if n.module_path in {"left", "left_norm"}
    }
    selected.update(
        n.canonical_id
        for n in context.trace_graph.compute_nodes
        if n.op_type == "relu" and any(p in selected for p in n.parents)
    )
    runtime = context.materialize(context.analyze(GraphPartition(frozenset(selected))))
    with pytest.raises(SplitUnsupportedError) as caught:
        runtime.run_training_prefix(torch.empty(1, 2, 3, 3, dtype=torch.float64, device="meta"))
    diagnostic = caught.value._partition_diagnostic
    assert diagnostic["category"] == "alias_provenance_unknown"
    assert diagnostic["source_alias_conflict"][2] == "unknown"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_native_dtype_passthrough_remains_precision_agnostic(dtype: torch.dtype) -> None:
    """FP64 oracles do not change the dtype chosen by runtime users."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).to(dtype=dtype)
    x = torch.ones(2, 4, dtype=dtype)
    context = prepare_graph(model, x)
    nodes = context.trace_graph.compute_nodes
    runtime = context.materialize(
        context.analyze(GraphPartition(frozenset({nodes[0].canonical_id})))
    )
    actual = runtime.replay(x)
    assert actual.dtype == dtype
    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, model(x))


def test_explicit_generator_calls_keep_their_native_order() -> None:
    """Per-call global RNG snapshots cannot justify reordering an explicit Generator."""

    class RandomBranches(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.Generator().manual_seed(10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Consume one shared native generator on two data-independent branches."""
            a = torch.rand((4,), generator=self.generator)
            b = torch.rand((4,), generator=self.generator)
            return x + a + b

    context = prepare_graph(RandomBranches(), torch.ones(1, 4, dtype=torch.float64))
    calls = [n for n in context.trace_graph.compute_nodes if n.op_type == "rand"]
    assert len(calls) == 2
    with pytest.raises(SplitUnsupportedError) as caught:
        context.analyze(GraphPartition(frozenset({calls[1].canonical_id})))
    assert caught.value.context.reason == "partition_effect_order_unproven"
