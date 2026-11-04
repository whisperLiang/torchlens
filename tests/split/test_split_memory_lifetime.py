"""Split memory reductions preserve outputs, aliases, gradients and probe evidence."""

from __future__ import annotations

import weakref
from typing import Any

import pytest
import torch
from dag_helpers import close_call_ids, smallest_transfer_partitions

from torchlens.split import SplitFeatures, pipeline, prepare_graph
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError


class Chain(torch.nn.Module):
    """Produce many independent tensor allocations with a narrow live frontier."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a chain whose dead outputs need not survive the next call."""

        for _ in range(24):
            x = torch.sin(x + 0.1)
        return x


@pytest.mark.parametrize("segment_name", ["prefix", "suffix"])
def test_replay_releases_dead_tensor_objects_within_segment(
    segment_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full segment retains only the live frontier, not every past allocation."""

    x = torch.randn(2, 8)
    model = Chain()
    with torch.no_grad():
        seed_graph = prepare_graph(model, x, features=SplitFeatures(retain_trace=True))
        seed_dag = seed_graph.compute_dag()
        seed_partition = seed_dag.to_partition(
            close_call_ids(
                seed_dag,
                frozenset(
                    call.call_id
                    for call in seed_dag.nodes[: (len(seed_dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        seed = seed_graph.materialize(seed_graph.analyze(seed_partition))
    dag = seed.compute_dag()
    partition = dag.to_partition(
        (call.call_id for call in dag.nodes) if segment_name == "prefix" else ()
    )
    runtime = seed.repartition(partition)
    segment = getattr(runtime.segments, segment_name)
    execute = segment._execute_func
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    max_live = 0

    def observe(node: Any, args: Any, kwargs: Any) -> Any:
        """Observe actual Python tensor lifetime at kernel entry."""

        nonlocal max_live
        max_live = max(max_live, sum(ref() is not None for ref in references))
        result = execute(node, args, kwargs)
        if isinstance(result, torch.Tensor):
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(segment, "_execute_func", observe)
    with torch.no_grad():
        actual = runtime.replay(x)
        torch.testing.assert_close(actual, model(x))
    assert len(references) >= 24
    assert max_live <= 2
    # The optimization must not remove the source trace's saved diagnostics.
    assert all(op.out is not None for op in seed.trace)
    seed.trace.cleanup()


class AliasedBranches(torch.nn.Module):
    """Mix multi-output views, an in-place write and an early output branch."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Require aliased storage and a completed output across later cuts."""

        early = torch.sin(x)
        base = x.clone()
        left, right = base.chunk(2, dim=1)
        left.add_(2)
        middle = right * 3
        return {"early": early, "base": base, "result": left + middle}


def test_every_cut_keeps_multioutput_aliases_and_early_output() -> None:
    """Releasing overlay references cannot change a view mutation or output tree."""

    model = AliasedBranches()
    x = torch.randn(2, 8)
    with torch.no_grad():
        seed_graph = prepare_graph(model, x)
        seed_dag = seed_graph.compute_dag()
        seed_partition = seed_dag.to_partition(
            close_call_ids(
                seed_dag,
                frozenset(
                    call.call_id
                    for call in seed_dag.nodes[: (len(seed_dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        seed = seed_graph.materialize(seed_graph.analyze(seed_partition))
        expected = model(x)
        for partition in smallest_transfer_partitions(seed):
            runtime = seed.repartition(partition)
            actual = runtime.replay(x)
            assert actual.keys() == expected.keys()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])
    assert not seed.retains_trace


def test_graph_connected_replay_still_computes_input_gradients() -> None:
    """Autograd owns required saved tensors after the overlay drops its references."""

    model = Chain()
    x = torch.randn(2, 8, requires_grad=True)
    expected = model(x)
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    graph = prepare_graph(model, x, features=SplitFeatures(training=True))
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    boundary = runtime.run_training_prefix(x)
    actual = runtime.run_suffix(boundary)
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    runtime.trace.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_replay_peak_tracks_frontier_not_chain_length() -> None:
    """Allocator peak measures real reclaimed storage, not just dead tensor objects."""

    x = torch.ones(2, 32768, device="cuda")
    with torch.no_grad():
        seed_graph = prepare_graph(Chain(), x, features=SplitFeatures(batch_axes={}))
        seed_dag = seed_graph.compute_dag()
        seed_partition = seed_dag.to_partition(
            close_call_ids(
                seed_dag,
                frozenset(
                    call.call_id
                    for call in seed_dag.nodes[: (len(seed_dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        seed = seed_graph.materialize(seed_graph.analyze(seed_partition))
        first = seed.trace_graph.compute_nodes[0]
        runtime = seed.repartition(
            seed.compute_dag().to_partition(
                close_call_ids(
                    seed.compute_dag(),
                    frozenset(
                        source
                        for source, target, _ in seed.compute_dag().data_edges
                        if target
                        in frozenset(
                            {
                                seed.trace_graph.replay_call_by_output_id[
                                    str(first.canonical_id)
                                ].call_id
                            }
                        )
                    ),
                )
            )
        )
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = runtime.replay(x)
        torch.cuda.synchronize()
        replay_peak = torch.cuda.max_memory_allocated() - baseline
        # Forty-eight independent allocations used to survive together.
        # Allow four input-sized buffers for the frontier and allocator rounding.
        assert replay_peak < 4 * x.numel() * x.element_size()
        torch.testing.assert_close(actual, Chain()(x))
    assert not seed.retains_trace


def test_probe_releases_witness_capture_before_generated_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lean B=2 capture is disposed before replay while its evidence survives."""

    capture = pipeline.capture_model
    build_segments = pipeline.execute_split_runtime
    witnesses: list[weakref.ReferenceType[Any]] = []
    checked = False

    def observe_capture(model: Any, inputs: Any, spec: Any, **kwargs: Any) -> Any:
        """Observe the production lean witness without retaining its capture."""

        result = capture(model, inputs, spec, **kwargs)
        if inputs[0].shape[0] == 2:
            assert kwargs["shape_witness"] is True
            witnesses.append(weakref.ref(result))
        return result

    def observe_segments(*args: Any, **kwargs: Any) -> Any:
        """Check the temporary capture's storage is no longer held at replay entry."""

        nonlocal checked
        if witnesses:
            assert all(ref() is None or ref()._tl_cleaned_up for ref in witnesses)
            checked = True
        return build_segments(*args, **kwargs)

    monkeypatch.setattr(pipeline, "capture_model", observe_capture)
    monkeypatch.setattr(pipeline, "execute_split_runtime", observe_segments)
    with torch.no_grad():
        graph = prepare_graph(Chain(), torch.ones(4, 8), features=SplitFeatures(retain_trace=True))
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(
                dag,
                frozenset(
                    call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        runtime = graph.materialize(graph.analyze(partition))
    assert checked
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
    assert all(op.out is not None for op in runtime.trace)
    with torch.no_grad():
        torch.testing.assert_close(runtime.replay(torch.ones(3, 8)), Chain()(torch.ones(3, 8)))
    runtime.trace.cleanup()


def test_shape_witness_omits_values_without_changing_retained_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the disposable B=2 trace omits output and argument snapshots."""

    capture = pipeline.capture_model
    observations: list[tuple[int, int, int]] = []

    def observe(model: Any, inputs: Any, spec: Any, **kwargs: Any) -> Any:
        """Count payloads before the witness is cleaned up."""

        result = capture(model, inputs, spec, **kwargs)
        observations.append(
            (
                int(inputs[0].shape[0]),
                sum(op.out is not None for op in result),
                sum(bool(op.has_saved_args) for op in result),
            )
        )
        return result

    monkeypatch.setattr(pipeline, "capture_model", observe)
    with torch.no_grad():
        graph = prepare_graph(Chain(), torch.ones(4, 8), features=SplitFeatures(retain_trace=True))
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(
                dag,
                frozenset(
                    call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        runtime = graph.materialize(graph.analyze(partition))
    assert len(observations) == 2
    assert observations[0][0] == 1
    assert observations[0][1] > 0 and observations[0][2] > 0
    assert observations[1] == (2, 0, 0)
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
    runtime.trace.cleanup()


def test_failed_shape_witness_never_recaptures_with_full_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed lean witness restricts replay instead of allocating a full archive."""

    class MutatingModel(torch.nn.Module):
        """Mutate a probe input before a stochastic operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Exercise failed-probe input mutation and stochastic state restoration."""

            x.add_(1)
            return x * torch.rand_like(x)

    normalize = pipeline.split_graph_from_trace
    attempts: list[bool] = []

    def require_saved_arguments(trace: Any) -> Any:
        """Simulate a witness needing the historical saved-scalar repair path."""

        attempts.append(trace.save_arg_values)
        if not trace.save_arg_values:
            raise SplitUnsupportedError("scalar parent requires saved arguments")
        return normalize(trace)

    monkeypatch.setattr(pipeline, "split_graph_from_trace", require_saved_arguments)
    inputs = torch.ones(4, 8)
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        graph = prepare_graph(MutatingModel(), inputs)
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(
                dag,
                frozenset(
                    call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]
                ),
            )
        )
        runtime = graph.materialize(graph.analyze(partition))
    assert attempts == [True, False]
    assert runtime.batch_validation["status"] == "failed", runtime.batch_validation
    assert "scalar parent requires saved arguments" in runtime.batch_validation["reason"]
    torch.testing.assert_close(inputs, torch.ones(4, 8))
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert runtime.replay(torch.ones(1, 8)).shape == (1, 8)
    with pytest.raises(SplitBoundaryError, match="probe did not pass"):
        runtime.replay(torch.ones(3, 8))
    assert not runtime.retains_trace
