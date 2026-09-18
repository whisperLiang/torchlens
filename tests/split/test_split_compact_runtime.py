"""Compact Torch split execution must not retain a diagnostic activation archive."""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.split import PlacementPlan, SplitFeatures, SplitRequest, pipeline
from torchlens.split.errors import SplitUnsupportedError


def _request(
    *,
    retain_trace: bool | None = None,
    training: bool = False,
    live_param_sources: bool | None = None,
    batch_axes: dict[str, int] | None = None,
) -> SplitRequest:
    """Build a typed request without opting existing fixtures into retention."""

    return SplitRequest(
        point=tl.split.percent(50),
        features=SplitFeatures(
            retain_trace=retain_trace,
            training=training,
            live_param_sources=live_param_sources,
            batch_axes=batch_axes,
        ),
    )


class AllocationChain(torch.nn.Module):
    """Use only executed intermediate tensors, with no tensor-valued constants."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Allocate independent values whose capture snapshots are disposable."""

        for _ in range(6):
            x = torch.sin(x + 0.1)
        return x


@pytest.mark.parametrize("retain_trace", [None, False])
def test_inference_defaults_to_compact_recuttable_runtime(retain_trace: bool | None) -> None:
    """Recutting and same-device placement do not resurrect a diagnostic Trace."""

    model = AllocationChain()
    x = torch.randn(3, 4)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, _request(retain_trace=retain_trace))
        assert not runtime.retains_trace
        assert runtime._trace is None
        with pytest.raises(SplitUnsupportedError, match="retain_trace=True"):
            _ = runtime.trace
        assert runtime.batch_validation["status"] == "passed"
        expected = model(x)
        for changed in (
            runtime,
            runtime.at(tl.split.before(runtime.trace_graph.compute_nodes[0].canonical_id)),
            runtime.at(tl.split.after(runtime.trace_graph.compute_nodes[-1].canonical_id)),
            runtime.with_placement(PlacementPlan.on("cpu")),
        ):
            assert not changed.retains_trace
            assert changed._trace is None
            assert changed.trace_graph is runtime.trace_graph
            torch.testing.assert_close(changed.replay(x), expected)


def test_compact_runtime_releases_original_trace_and_intermediate_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An executable graph must not pin the Trace through its op or parameter metadata."""

    capture = pipeline.capture_model
    traces: list[weakref.ReferenceType[Any]] = []
    payloads: list[weakref.ReferenceType[torch.Tensor]] = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        """Observe captured objects without retaining any of them strongly."""

        trace = capture(*args, **kwargs)
        traces.append(weakref.ref(trace))
        payloads.extend(
            weakref.ref(op.out)
            for op in trace
            if not op.is_input and isinstance(op.out, torch.Tensor)
        )
        return trace

    monkeypatch.setattr(pipeline, "capture_model", observe)
    model = AllocationChain()
    x = torch.randn(3, 4)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, _request(batch_axes={}))
    gc.collect()
    assert traces and all(reference() is None for reference in traces)
    assert all(reference() is None for reference in payloads)
    assert all(getattr(node.op, "out", None) is None for node in runtime.trace_graph.nodes)
    with torch.no_grad():
        torch.testing.assert_close(runtime.replay(x), model(x))


def test_diagnostic_retention_preserves_trace_values_and_recut_identity() -> None:
    """Explicit retention keeps the original diagnostic surface available."""

    model = AllocationChain()
    x = torch.randn(3, 4)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, _request(retain_trace=True))
        assert runtime.retains_trace
        assert runtime.trace is runtime._trace
        assert all(op.out is not None for op in runtime.trace)
        recut = runtime.at(tl.split.before(runtime.trace_graph.compute_nodes[0].canonical_id))
        assert recut.trace is runtime.trace
        torch.testing.assert_close(recut.replay(x), model(x))
    runtime.trace.cleanup()


def test_compact_replay_and_recut_restore_captured_cpu_autocast() -> None:
    """Replay outside autocast retains the captured bfloat16 dtype and values."""

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))
    x = torch.randn(2, 4)
    with torch.no_grad():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = model(x)
            runtime = tl.split.prepare(model, x, _request(batch_axes={}))
        assert expected.dtype == torch.bfloat16
        assert not torch.is_autocast_enabled("cpu")
        assert not runtime.retains_trace
        assert any(
            node.op.func_autocast_state["cpu"]["enabled"]
            for node in runtime.trace_graph.compute_nodes
        )
        first, last = runtime.trace_graph.compute_nodes[0], runtime.trace_graph.compute_nodes[-1]
        for changed in (
            runtime,
            runtime.at(tl.split.before(first.canonical_id)),
            runtime.at(tl.split.after(last.canonical_id)),
        ):
            actual = changed.replay(x)
            assert actual.dtype == torch.bfloat16
            torch.testing.assert_close(actual, expected)
            assert not torch.is_autocast_enabled("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compact_preparation_releases_cuda_activation_archive() -> None:
    """Allocator bytes prove compact preparation releases physical CUDA storage."""

    model = AllocationChain()
    x = torch.ones(2, 32768, device="cuda")
    retained: dict[bool, int] = {}
    with torch.no_grad():
        for retain_trace in (True, False):
            gc.collect()
            baseline = torch.cuda.memory_allocated()
            runtime = tl.split.prepare(model, x, _request(retain_trace=retain_trace, batch_axes={}))
            gc.collect()
            torch.cuda.synchronize()
            retained[retain_trace] = torch.cuda.memory_allocated() - baseline
            actual = runtime.replay(x)
            torch.testing.assert_close(actual, model(x))
            if runtime.retains_trace:
                runtime.trace.cleanup()
            del actual, runtime
            gc.collect()
    assert retained[True] >= 12 * x.numel() * x.element_size()
    assert retained[False] < retained[True] // 4


class ConstantSources(torch.nn.Module):
    """Combine registered state, plain tensors and factory literals."""

    def __init__(self) -> None:
        """Keep constant sources independent of ordinary activation snapshots."""

        super().__init__()
        self.register_buffer("offset", torch.tensor([[1.0, 2.0, 3.0, 4.0]]), persistent=False)
        self.literal = torch.tensor([0.25, 0.5, 0.75, 1.0])

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """Reassemble nested outputs while retaining every genuine state source."""

        hidden = x * 0.5 + self.offset
        shifted = hidden + self.literal
        result = shifted + torch.tensor([0.5, 1.0, 1.5, 2.0], device=x.device)
        return {"pair": (result, hidden), "metadata": [None, 7, "fixed"]}


@pytest.mark.parametrize("batch", [1, 3])
def test_compact_replay_keeps_required_state_constants_and_nested_outputs(batch: int) -> None:
    """Removing snapshots must not remove factory literals, buffers or output metadata."""

    def assert_output(actual: Any, expected: Any) -> None:
        """Compare tensor leaves numerically and string/container metadata exactly."""

        assert type(actual) is type(expected)
        assert tuple(actual) == tuple(expected)
        assert type(actual["pair"]) is type(expected["pair"])
        torch.testing.assert_close(actual["pair"], expected["pair"])
        assert type(actual["metadata"]) is type(expected["metadata"])
        assert actual["metadata"] == expected["metadata"]

    model = ConstantSources().eval()
    x = torch.randn(3, 4)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, _request())
        diagnostic = tl.split.prepare(model, x, _request(retain_trace=True))
        actual_input = x[:batch]
        expected = model(actual_input)
        assert_output(runtime.replay(actual_input), expected)
        checked = 0
        for node in runtime.trace_graph.compute_nodes:
            for point in (tl.split.before(node.canonical_id), tl.split.after(node.canonical_id)):
                try:
                    reference = diagnostic.at(point)
                except SplitUnsupportedError as exc:
                    # Plain tensor sources can have no live buffer handle.
                    # The established recut guard then rejects merging the
                    # inference and training-prefix states in either mode.
                    with pytest.raises(SplitUnsupportedError) as compact_error:
                        runtime.at(point)
                    assert compact_error.value.context == exc.context
                    continue
                actual = runtime.at(point).replay(actual_input)
                assert_output(actual, expected)
                assert_output(actual, reference.replay(actual_input))
                checked += 1
        assert checked > 0
        diagnostic.trace.cleanup()


class AliasedResidual(torch.nn.Module):
    """Require an early residual, multiple views and an observable in-place update."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exercise every boundary across shared storage and a long-lived residual."""

        residual = torch.sin(x)
        base = x.clone()
        left, right = base.chunk(2, dim=1)
        left.add_(2)
        mixed = left + right * 3
        return {"residual": residual + base, "base": base, "mixed": mixed}


def test_compact_every_cut_preserves_residuals_views_and_inplace_updates() -> None:
    """A boundary carries the complete dependency frontier rather than one chosen output."""

    model = AliasedResidual()
    x = torch.randn(3, 8)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, _request())
        expected = model(x)
        for node in runtime.trace_graph.compute_nodes:
            for point in (tl.split.before(node.canonical_id), tl.split.after(node.canonical_id)):
                torch.testing.assert_close(runtime.at(point).replay(x), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compact_placement_preserves_live_parameter_identity_and_model_device(device: str) -> None:
    """Placed replicas remain independent while unplaced live handles retain identity."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous compact placement.")
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    x = torch.randn(3, 4)
    with torch.no_grad():
        runtime = tl.split.prepare(model, x, _request(live_param_sources=True))
        parameters = [
            reference.handle for node in runtime.trace_graph.nodes for reference in node.param_refs
        ]
        assert {id(value) for value in parameters} == {id(value) for value in model.parameters()}
        model[2].bias.add_(0.5)
        torch.testing.assert_close(runtime.replay(x), model(x))
        placed = runtime.with_placement(PlacementPlan.across("cpu", device))
        actual = placed.replay(x)
        assert actual.device == torch.device(device)
        torch.testing.assert_close(actual.cpu(), model(x))
    assert placed._trace is None
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.parametrize("retain_trace", [None, True])
def test_training_automatically_retains_trace_and_preserves_input_and_parameter_gradients(
    retain_trace: bool | None,
) -> None:
    """Compact inference defaults cannot silently detach an explicitly trainable split."""

    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    x = torch.randn(3, 4, requires_grad=True)
    expected = model(x)
    targets = (x, *model.parameters())
    expected_gradients = torch.autograd.grad(expected.sum(), targets)
    runtime = tl.split.prepare(model, x, _request(training=True, retain_trace=retain_trace))
    assert runtime.retains_trace
    assert runtime.trace is not None
    boundary = runtime.run_training_prefix(x)
    actual = runtime.run_suffix(boundary)
    actual_gradients = torch.autograd.grad(actual.sum(), targets)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    runtime.trace.cleanup()


def test_training_with_explicit_compact_retention_refuses() -> None:
    """An explicitly contradictory retention request must never silently discard gradients."""

    with pytest.raises(SplitUnsupportedError):
        tl.split.prepare(
            AllocationChain(),
            torch.randn(3, 4),
            _request(training=True, retain_trace=False),
        )


@pytest.mark.parametrize("value", [0, 1, "true", {}, []])
def test_trace_retention_flag_rejects_non_boolean_values(value: Any) -> None:
    """Retention is a closed boolean policy, not an arbitrary truthy object."""

    with pytest.raises(TypeError, match="retain_trace"):
        SplitFeatures(retain_trace=value)
