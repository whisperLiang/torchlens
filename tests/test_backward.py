"""Smoke tests for first-class backward-pass capture."""

import warnings
from types import MethodType
from unittest import mock

import torch
from torch import nn
import pytest

import torchlens as tl
import torchlens.validation as tl_validation
import torchlens.validation.backward as backward_validation
import torchlens.validation.consolidated as consolidated_validation
from torchlens.data_classes.grad_fn import GradFn
from torchlens.ir.events import BackwardPassStart, OpGradObserved
from torchlens.options import CaptureOptions, SaveOptions

_NO_GRAD_AUTOGRAD_ERROR = "element 0 of tensors does not require grad and does not have a grad_fn"


class _TinyBackwardModel(nn.Module):
    """Small MLP with view op coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        hidden = torch.relu(self.fc1(x))
        viewed = hidden.view(hidden.shape[0], 4)
        return self.fc2(viewed)


class _DoubleFn(torch.autograd.Function):
    """Custom autograd function for grad_fn_handle classification tests."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Return doubled input."""
        return x * 2

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return doubled upstream grad."""
        return grad * 2


class _CustomModel(nn.Module):
    """Model using a custom autograd Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return _DoubleFn.apply(x).sum()


class _SquareFunction(torch.autograd.Function):
    """Custom autograd function with non-trivial gradient."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Square the input tensor."""

        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return the analytical square gradient."""

        (x,) = ctx.saved_tensors
        return 2 * x * grad


class _CustomParamModel(nn.Module):
    """Parameterized model that routes through a custom autograd Function."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the custom function on a parameterized activation."""

        return self.output(_SquareFunction.apply(x @ self.weight)).sum()


class _WeightTiedModel(nn.Module):
    """Model that applies one module twice."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a shared layer twice."""

        return self.linear(torch.relu(self.linear(x))).sum()


class _DropoutBackwardModel(nn.Module):
    """Dropout model for seeded backward validation."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout and a linear layer."""

        return self.linear(self.dropout(x)).sum()


class _BatchNormBackwardModel(nn.Module):
    """BatchNorm model for state restoration checks."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm and a linear layer."""

        return self.linear(self.bn(x)).sum()


def _logged_model(
    *,
    layers_to_save: str | list[str] | None = "all",
    save_grads: str | list[str] | None = "all",
) -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Create a logged tiny model.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, tl.Trace]
        Model, input tensor, and model log.
    """
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save=layers_to_save, save_grads=save_grads),
    )
    return model, x, trace


def _output_loss(trace: tl.Trace) -> torch.Tensor:
    """Return scalar sum loss from the logged output out."""
    return trace[trace.output_layers[0]].out.sum()


def _saved_relu(trace: tl.Trace) -> torch.Tensor:
    """Return the selectively retained ReLU activation from a trace."""

    return next(op.out for op in trace if op.has_saved_activation and op.func_name == "relu")


def test_detached_saved_activation_backward_names_backward_ready_remedy() -> None:
    """A TorchLens-detached retained activation gives targeted backward guidance."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, save=tl.func("relu"))
    saved = _saved_relu(trace)

    with pytest.raises(RuntimeError, match=r"Re-trace with backward_ready=True") as exc_info:
        saved.sum().backward()

    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value).startswith(_NO_GRAD_AUTOGRAD_ERROR)
    assert x.grad is None


def test_unrelated_detached_tensor_backward_keeps_native_error() -> None:
    """An unrelated user-detached tensor keeps the unmodified PyTorch error."""

    detached = torch.randn(3, requires_grad=True).detach()

    with pytest.raises(RuntimeError) as exc_info:
        detached.sum().backward()

    assert str(exc_info.value) == _NO_GRAD_AUTOGRAD_ERROR

    trace = tl.trace(_TinyBackwardModel(), torch.randn(2, 3), save=tl.func("relu"))
    explicitly_detached = _saved_relu(trace).detach()
    with pytest.raises(RuntimeError) as marked_exc_info:
        explicitly_detached.sum().backward()

    assert str(marked_exc_info.value) == _NO_GRAD_AUTOGRAD_ERROR

    metadata_only = torch.randn_like(_saved_relu(trace)).view_as(_saved_relu(trace))
    with pytest.raises(RuntimeError) as metadata_exc_info:
        metadata_only.sum().backward()

    assert str(metadata_exc_info.value) == _NO_GRAD_AUTOGRAD_ERROR


def test_connected_output_and_backward_ready_saved_activation_are_unchanged() -> None:
    """Connected model outputs and backward-ready retained activations still backpropagate."""

    output_model = _TinyBackwardModel()
    output_x = torch.randn(2, 3, requires_grad=True)
    output_trace = tl.trace(output_model, output_x)
    output = output_trace.output_ops[0].out
    output.sum().backward()
    assert output_x.grad is not None

    ready_model = _TinyBackwardModel()
    ready_x = torch.randn(2, 3, requires_grad=True)
    ready_trace = tl.trace(
        ready_model,
        ready_x,
        save=tl.func("relu"),
        backward_ready=True,
    )
    ready_saved = _saved_relu(ready_trace)
    ready_saved.sum().backward()
    assert ready_x.grad is not None


def test_detached_log_backward_does_not_poison_later_capture() -> None:
    """Reject detached losses without leaking capture state or a start event."""
    model = nn.Linear(2, 1)
    trace = tl.trace(model, torch.ones(1, 2))

    with pytest.raises(ValueError, match="loss has no grad_fn / is detached"):
        trace.log_backward(torch.tensor(1.0))

    assert not any(isinstance(event, BackwardPassStart) for event in trace.backward_events)
    fresh_trace = tl.trace(model, torch.ones(1, 2))
    assert fresh_trace.output_layers

    recording = tl.record(model, torch.ones(1, 2), save=tl.func("linear"))
    sparse_trace = recording.to_trace()
    detached_output = sparse_trace[sparse_trace.output_layers[0]].out
    assert detached_output is not None
    assert detached_output.grad_fn is None

    with pytest.raises(ValueError, match="loss has no grad_fn / is detached"):
        sparse_trace.log_backward(detached_output)

    assert not any(isinstance(event, BackwardPassStart) for event in sparse_trace.backward_events)
    assert tl.trace(model, torch.ones(1, 2)).output_layers


@pytest.mark.smoke
def test_log_backward_captures_per_layer_grads() -> None:
    """log_backward captures saved per-layer grads."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.has_gradients
    assert len(trace.saved_grad_ops) > 0
    assert all(trace[label].grad is not None for label in trace.saved_grad_ops.keys())


@pytest.mark.smoke
def test_recording_backward_context_manager() -> None:
    """recording_backward accumulates multiple backward calls."""
    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        (loss * 2).backward()
    assert trace.num_backward_passes == 2


def _backward_projection_snapshot(trace: tl.Trace) -> dict:
    """Return a value-level snapshot of every backward projection surface."""

    grad_fns = {
        object_id: (
            grad_fn.label,
            grad_fn.type,
            grad_fn.type_index,
            grad_fn.ordinal_index,
            grad_fn.step_index,
            grad_fn.order,
            grad_fn.has_op,
            grad_fn.op_label,
            list(grad_fn.children),
            list(grad_fn.parents),
            grad_fn.module_address,
            grad_fn.module_membership_source,
            list(grad_fn.next_grad_fn_ids),
            sorted(grad_fn.calls),
        )
        for object_id, grad_fn in trace.grad_fn_logs.items()
    }
    calls = {
        (object_id, ordinal): (call.label, call.backward_pass_index)
        for object_id, grad_fn in trace.grad_fn_logs.items()
        for ordinal, call in grad_fn.calls.items()
    }
    passes = {
        pass_index: (
            record.trigger,
            record.status,
            record.order,
            len(record.grad_fn_calls),
            record.order_attribution_coverage,
        )
        for pass_index, record in trace.backward_pass_logs.items()
    }
    op_grads = {
        op.layer_label: (
            [
                (record.backward_pass_index, record.grad is not None, record.shape)
                for record in op._slot("_grad_records")
            ],
            int(op.gradient_memory),
        )
        for op in trace.layer_list
        if getattr(op, "has_grad", False)
    }
    param_grads = {
        address: [
            (record.ordinal, record.backward_pass_index, record.shape, record.memory)
            for record in param_log._grad_records
        ]
        for address, param_log in trace.param_logs.items()
        if param_log._grad_records
    }
    return {
        "grad_fns": grad_fns,
        "calls": calls,
        "passes": passes,
        "op_grads": op_grads,
        "param_grads": param_grads,
        "order": list(trace.grad_fn_order),
        "num_passes": trace.num_backward_passes,
        "num_calls": trace.num_saved_grad_fn_calls,
        "num_fns": trace.num_saved_grad_fns,
        "saved_labels": set(trace._saved_grad_labels),
        "total_grad_mem": int(trace.total_gradient_memory),
        "total_bwd_mem": int(trace.total_backward_memory),
    }


@pytest.mark.smoke
def test_backward_reprojection_folds_incrementally() -> None:
    """Repeated passes over an unchanged graph fold O(tail), not full rebuilds."""
    from torchlens.backends.torch import backward as backward_mod

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    full_rebuild_sizes: list[int] = []
    real_impl = backward_mod._materialize_backward_projections_impl

    def counting_impl(trace_arg: tl.Trace, events: list) -> None:
        full_rebuild_sizes.append(len(events))
        real_impl(trace_arg, events)

    with mock.patch.object(
        backward_mod, "_materialize_backward_projections_impl", counting_impl
    ):
        with trace.recording_backward():
            loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            loss.backward()
    assert trace.num_backward_passes == 3
    assert len(full_rebuild_sizes) == 1, (
        "later same-graph passes must fold incrementally, not rebuild: "
        f"{full_rebuild_sizes}"
    )

    incremental_snapshot = _backward_projection_snapshot(trace)
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)
    assert _backward_projection_snapshot(trace) == incremental_snapshot


@pytest.mark.smoke
def test_backward_events_share_one_monotonic_seq_domain() -> None:
    """Every backward event kind carries one writer-stamped monotonic seq."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd as _End

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    events = _ensure_backward_event_stream(trace).backward_events
    seqs = [event.seq for event in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    assert all(seq > 0 for seq in seqs)
    start = next(e for e in events if isinstance(e, BackwardPassStart))
    end = next(e for e in events if isinstance(e, _End))
    assert start.seq < end.seq
    for event in events:
        if getattr(event, "pass_index", None) == 1 and event is not start and event is not end:
            assert start.seq < event.seq < end.seq


def _invariant_check(trace: tl.Trace) -> None:
    """Run the backward event-flow invariant directly."""
    from torchlens.validation.invariants import _check_backward_event_flow_invariants

    _check_backward_event_flow_invariants(trace, "backward_graph_invariants")


@pytest.mark.smoke
def test_backward_seq_invariants_fire_on_planted_mutations() -> None:
    """Each rewritten exact-seq assertion still fails on a planted misorder."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd as _End
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    events = _ensure_backward_event_stream(trace).backward_events
    _invariant_check(trace)  # positive control: the real stream passes

    start = next(e for e in events if isinstance(e, BackwardPassStart))
    end = next(e for e in events if isinstance(e, _End))
    op_grad = next(e for e in events if isinstance(e, OpGradObserved))

    original_op_grad_seq = op_grad.seq
    object.__setattr__(op_grad, "seq", end.seq + 1)
    with pytest.raises(MetadataInvariantError, match="follows its pass|unique and monotonic"):
        _invariant_check(trace)
    object.__setattr__(op_grad, "seq", start.seq - 1 if start.seq > 1 else 0)
    with pytest.raises(MetadataInvariantError, match="precedes its pass|unique and monotonic"):
        _invariant_check(trace)
    object.__setattr__(op_grad, "seq", original_op_grad_seq)
    _invariant_check(trace)

    original_end_seq = end.seq
    object.__setattr__(end, "seq", start.seq)
    with pytest.raises(MetadataInvariantError, match="unique and monotonic|does not"):
        _invariant_check(trace)
    object.__setattr__(end, "seq", original_end_seq)
    _invariant_check(trace)

    original_start_seq = start.seq
    object.__setattr__(start, "seq", original_end_seq + 5)
    with pytest.raises(MetadataInvariantError):
        _invariant_check(trace)
    object.__setattr__(start, "seq", original_start_seq)
    _invariant_check(trace)

    # A monotonic-but-misbracketed stream (End reordered before the pass's
    # facts, all seqs renumbered in list order) must fail on bracketing
    # alone, proving the exact-bracket assertion is independently armed.
    original_order = list(events)
    original_seqs = [event.seq for event in events]
    events.remove(end)
    events.insert(events.index(start) + 1, end)
    for renumbered_seq, event in enumerate(events, start=1):
        object.__setattr__(event, "seq", renumbered_seq)
    with pytest.raises(MetadataInvariantError, match="follows its pass"):
        _invariant_check(trace)
    events[:] = original_order
    for original_seq, event in zip(original_seqs, events):
        object.__setattr__(event, "seq", original_seq)
    _invariant_check(trace)


@pytest.mark.smoke
def test_backward_capture_refuses_missing_event_stream() -> None:
    """A trace that lost its event stream gets a typed refusal, not a silent buffer."""
    from torchlens._errors import BackwardStreamUnavailableError
    from torchlens.backends.torch.backward import _ensure_backward_event_stream

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    del trace._capture_events
    assert trace.event_stream is None
    with pytest.raises(BackwardStreamUnavailableError):
        _ensure_backward_event_stream(trace)
    with pytest.raises(BackwardStreamUnavailableError):
        trace.log_backward(loss)


@pytest.mark.smoke
def test_param_gradients_enter_the_backward_event_stream() -> None:
    """Every recorded AccumulateGrad increment has a ParamGradObserved event."""
    from torchlens.ir.events import ParamGradObserved

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))

    param_events = [
        event for event in trace.backward_events if isinstance(event, ParamGradObserved)
    ]
    assert param_events
    event_records = {
        (event.param_address, event.pass_index) for event in param_events
    }
    projected_records = {
        (address, record.backward_pass_index)
        for address, param_log in trace.param_logs.items()
        for record in param_log._grad_records
    }
    assert projected_records
    assert event_records == projected_records
    for event in param_events:
        assert event.payload_ref is not None
        assert event.shape is not None
        assert event.memory
        assert event.seq > 0


@pytest.mark.smoke
def test_replay_fork_does_not_inherit_gradient_state() -> None:
    """A replay fork starts with no captured gradient state; the source keeps its own."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_grads="all"),
        intervention_ready=True,
        backward_ready=True,
    )
    trace.log_backward(_output_loss(trace))
    assert trace.has_gradients
    assert trace._saved_grad_labels
    assert any(param_log._grad_records for param_log in trace.param_logs.values())

    def _identity_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        return out

    fork = trace.replay(hooks={tl.func("relu"): _identity_hook}, differentiable=True)

    assert fork.has_gradients is False
    assert fork._saved_grad_labels == set()
    assert not any(param_log._grad_records for param_log in fork.param_logs.values())
    assert int(fork.total_param_gradient_memory) == 0
    assert fork.num_backward_passes == 0

    assert trace.has_gradients
    assert trace._saved_grad_labels
    assert any(param_log._grad_records for param_log in trace.param_logs.values())


@pytest.mark.smoke
def test_backward_reprojection_guard_survives_count_preserving_mutation() -> None:
    """A count-preserving event mutation still triggers reprojection."""
    from dataclasses import replace as dataclass_replace

    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.backward_pass_logs[1].status == "ok"

    stream = _ensure_backward_event_stream(trace)
    old_end = stream.backward_events[-1]
    assert isinstance(old_end, BackwardPassEnd)
    stream.backward_events.pop()
    stream.note_backward_event_removal()
    stream.append_backward(dataclass_replace(old_end, status="error"))

    trace._sync_backward_projection_if_needed()
    assert trace.backward_pass_logs[1].status == "error"


@pytest.mark.smoke
def test_recording_backward_delegates_foreign_graphs() -> None:
    """A backward on an unrelated graph inside the context never enters the trace."""
    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    foreign = torch.randn(3, 3, requires_grad=True)
    with trace.recording_backward():
        (foreign * 2).sum().backward()
        loss.backward()
    assert foreign.grad is not None
    assert trace.num_backward_passes == 1

    control_model, _control_x, control_trace = _logged_model()
    control_loss = _output_loss(control_trace)
    with control_trace.recording_backward():
        control_loss.backward()
    trace_labels = sorted(grad_fn.label for grad_fn in trace.grad_fn_logs.values())
    control_labels = sorted(grad_fn.label for grad_fn in control_trace.grad_fn_logs.values())
    assert trace_labels == control_labels


@pytest.mark.smoke
def test_recording_backward_foreign_only_block_stays_empty() -> None:
    """A foreign-only block records no passes and warns once, not silently."""
    _model, _x, trace = _logged_model()
    foreign = torch.randn(2, 2, requires_grad=True)
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        with trace.recording_backward():
            (foreign * foreign).sum().backward()
            (foreign * 2.0).sum().backward()
    unmatched_warnings = [
        record
        for record in warning_records
        if issubclass(record.category, RuntimeWarning)
        and "did not reach any grad-fn" in str(record.message)
    ]
    assert len(unmatched_warnings) == 1, "unmatched-backward warning must fire exactly once"
    assert foreign.grad is not None
    assert trace.num_backward_passes == 0
    assert len(trace.grad_fn_logs) == 0


@pytest.mark.smoke
def test_recording_backward_exit_preserves_interleaved_patch() -> None:
    """__exit__ never clobbers a Tensor.backward patch installed inside the block."""
    _model, _x, trace = _logged_model()

    def interloper_backward(tensor_self: torch.Tensor, *args: object, **kwargs: object) -> None:
        raise AssertionError("interloper should never run in this test")

    context = trace.recording_backward()
    context.__enter__()
    try:
        torch.Tensor.backward = interloper_backward  # type: ignore[assignment, method-assign]
        with pytest.warns(UserWarning, match="Tensor.backward"):
            context.__exit__(None, None, None)
        assert torch.Tensor.backward is interloper_backward
    finally:
        torch.Tensor.backward = context._original_backward  # type: ignore[assignment, method-assign]


@pytest.mark.smoke
def test_backward_graph_walk_includes_intervening_grad_fns() -> None:
    """The backward DAG includes grad_fns without forward Layer matches."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(not grad_fn_handle.has_op for grad_fn_handle in trace.grad_fn_logs.values())


def test_has_op_storage_field() -> None:
    """GradFn.has_op records whether a forward op was captured."""

    grad_fn_handle = GradFn(
        grad_fn_object_id=1,
        class_name="AddBackward0",
        class_qualname="torch.autograd.AddBackward0",
        is_custom=False,
        label="addbackward0_1_1",
        type="addbackward0",
        type_index=1,
        ordinal_index=1,
        step_index=1,
        has_op=True,
    )
    assert grad_fn_handle.has_op is True


@pytest.mark.smoke
def test_grad_fn_log_back_pointer() -> None:
    """Forward LayerLogs link to corresponding GradFnLogs by identity."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(layer.grad_fn is not None for layer in trace.layer_list)
    assert any(
        grad_fn_handle.op is not None and grad_fn_handle.op.grad_fn is grad_fn_handle
        for grad_fn_handle in trace.grad_fns
    )


@pytest.mark.smoke
def test_grad_fn_naming_and_indexing() -> None:
    """GradFn labels and accessor indexing mirror layer lookup patterns."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    first_grad_fn = trace.grad_fns[0]
    assert "_back_" in first_grad_fn.label
    assert first_grad_fn.label == first_grad_fn.label.lower()
    assert trace.grad_fns[first_grad_fn.label] is first_grad_fn
    assert trace.grad_fns[first_grad_fn.type] is first_grad_fn
    if first_grad_fn.num_calls:
        assert trace.grad_fns[f"{first_grad_fn.label}:1"] is first_grad_fn
        assert trace.grad_fn_calls[f"{first_grad_fn.label}:1"] is first_grad_fn.calls[0]
    assert list(trace.grad_fns)


@pytest.mark.smoke
def test_save_grads_true_captures_all_grads() -> None:
    """save_grads=True captures all gradients independent of layers_to_save."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=["relu"], save_grads=True))
    trace.log_backward(_output_loss(trace))
    assert trace.saved_grad_ops
    assert any("relu" in label for label in trace.saved_grad_ops.keys())
    assert any("linear" in label for label in trace.saved_grad_ops.keys())


@pytest.mark.smoke
def test_grads_to_save_independent_override() -> None:
    """save_grads selectors are independent from layers_to_save."""
    _model, _x, trace = _logged_model(layers_to_save="all", save_grads=["relu"])
    trace.log_backward(_output_loss(trace))
    assert trace.saved_grad_ops
    assert all("relu" in label for label in trace.saved_grad_ops.keys())


@pytest.mark.smoke
def test_auto_train_mode_when_backward_opted_in() -> None:
    """Explicit save_grads selectors auto-enable backward_ready."""
    _model, _x, trace = _logged_model()
    assert trace.backward_ready is True


@pytest.mark.smoke
def test_auto_train_mode_conflict_with_explicit_false() -> None:
    """Explicit backward_ready=False conflicts with backward capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="requires backward_ready=True"):
        tl.trace(model, x, capture=CaptureOptions(save_grads="all", backward_ready=False))


@pytest.mark.smoke
def test_grad_transform_applied() -> None:
    """grad_transform writes transformed grads separately."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(save_grads="all"),
        save=SaveOptions(grad_transform=lambda grad: torch.zeros_like(grad)),
    )
    trace.log_backward(_output_loss(trace))
    assert all(
        torch.equal(
            trace[label].transformed_grad,
            torch.zeros_like(trace[label].grad),
        )
        for label in trace.saved_grad_ops.keys()
    )


@pytest.mark.smoke
def test_flat_transform_kwargs_populate_transformed_payloads() -> None:
    """Flat activation_transform and grad_transform kwargs are applied."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.warns(DeprecationWarning):
        trace = tl.trace(
            model,
            x,
            activation_transform=lambda out: out.half(),
            grad_transform=lambda grad: grad.half(),
            save_grads=True,
            backward_ready=True,
        )
    relu_op = next(op for op in trace.ops if op.func_name == "relu")

    trace.log_backward(_output_loss(trace), retain_graph=True)

    assert relu_op.transformed_out is not None
    assert relu_op.transformed_out.dtype == torch.float16
    assert relu_op.transformed_grad is not None
    assert relu_op.transformed_grad.dtype == torch.float16


@pytest.mark.smoke
def test_module_log_grad_aggregation() -> None:
    """Module exposes aggregated grads for contained layers."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.modules["fc2"].grad is not None


@pytest.mark.smoke
def test_input_layer_grad_access() -> None:
    """Input layers expose saved grads after backward."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace[trace.input_layers[0]].grad is not None


@pytest.mark.smoke
def test_param_layer_grad_access() -> None:
    """Param grad metadata still works through the existing hook path."""
    model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(param_log.has_grad for param_log in trace.params)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.smoke
def test_custom_autograd_function_captured_with_is_custom_flag() -> None:
    """Custom autograd.Function grad_fns are captured and flagged."""
    model = _CustomModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=CaptureOptions(save_grads="all"))
    trace.log_backward(_output_loss(trace))
    assert any(grad_fn_handle.is_custom for grad_fn_handle in trace.grad_fn_logs.values())


@pytest.mark.smoke
def test_implicit_hook_firing_preserved() -> None:
    """Calling backward outside log_backward still populates Layer grads."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=CaptureOptions(save_grads=True))
    _output_loss(trace).backward()
    assert trace.saved_grad_ops


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:`layers_to_save` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:`random_seed` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:`save_grads` is deprecated:DeprecationWarning")
def test_validate_backward_pass_correct() -> None:
    """validate_backward_pass returns True for correct capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x)


def test_validate_backward_default_detects_corrupted_captured_op_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default backward validation must read captured ``Op.grad`` payloads."""

    original_log_backward = tl.Trace.log_backward
    corrupted_labels: list[str] = []

    def corrupt_captured_grad(trace: tl.Trace, *args: object, **kwargs: object) -> object:
        """Corrupt one captured module-output grad after backward logging."""

        result = original_log_backward(trace, *args, **kwargs)
        for op in trace.layer_list:
            grad = getattr(op, "grad", None)
            if not isinstance(grad, torch.Tensor) or not getattr(op, "modules", None):
                continue
            record = op.grads.for_pass(1)
            record.grad = torch.full_like(grad, 12345.0)
            corrupted_labels.append(op.label)
            break
        return result

    monkeypatch.setattr(tl.Trace, "log_backward", corrupt_captured_grad)

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert not tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert corrupted_labels


def test_validate_backward_pass_random_seed_kwarg_public_wrapper() -> None:
    """The public backward wrapper forwards ``random_seed`` into the dispatcher."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with mock.patch(
        "torchlens.validation.consolidated.validate_backward_pass",
        wraps=consolidated_validation.validate_backward_pass,
    ) as validator:
        with pytest.warns(DeprecationWarning):
            assert tl.validate_backward_pass(model, x, random_seed=42)
    assert validator.call_args is not None
    assert validator.call_args.kwargs["random_seed"] == 42


def test_validation_validate_backward_pass_random_seed_kwarg_user_funcs_path() -> None:
    """The validation subpackage path forwards ``random_seed`` to the source validator."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with mock.patch(
        "torchlens.validation.backward.validate_backward_pass",
        wraps=backward_validation.validate_backward_pass,
    ) as validator:
        assert tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert validator.call_args is not None
    assert validator.call_args.kwargs["random_seed"] == 42


def test_validate_backward_random_seed_deterministic() -> None:
    """A fixed seed produces repeatable backward validation outcomes."""

    torch.manual_seed(0)
    model = _DropoutBackwardModel().train()
    x = torch.randn(6, 3, requires_grad=True)
    first = tl_validation.validate_backward_pass(model, x, random_seed=42)
    second = tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert first is True
    assert second is first


def test_validate_backward_zero_grad_between_passes() -> None:
    """Backward validation zeros parameter grads before, between, and after passes."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    calls: list[bool | None] = []
    original_zero_grad = model.zero_grad

    def counted_zero_grad(self: nn.Module, set_to_none: bool | None = True) -> None:
        """Record zero_grad calls and delegate to the original method."""

        calls.append(set_to_none)
        original_zero_grad(set_to_none=set_to_none)

    model.zero_grad = MethodType(counted_zero_grad, model)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert calls == [True, True, True]


def test_validate_backward_dropout_train_mode_reproducible() -> None:
    """Seeded validation handles train-mode Dropout."""

    torch.manual_seed(0)
    model = _DropoutBackwardModel().train()
    x = torch.randn(6, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_batchnorm_train_mode_state_restored() -> None:
    """Seeded validation handles train-mode BatchNorm without leaking state."""

    torch.manual_seed(0)
    model = _BatchNormBackwardModel().train()
    x = torch.randn(6, 3, requires_grad=True)
    running_mean = model.bn.running_mean.detach().clone()
    running_var = model.bn.running_var.detach().clone()

    assert tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert torch.equal(model.bn.running_mean, running_mean)
    assert torch.equal(model.bn.running_var, running_var)


def test_validate_backward_train_mode_audit() -> None:
    """Backward validation preserves the model train/eval flag."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    for training in (True, False):
        model.train(training)
        assert tl_validation.validate_backward_pass(model, x, random_seed=42)
        assert model.training is training


def test_validate_backward_multipass_weight_tied() -> None:
    """Parameter-grad comparison handles a shared module used twice."""

    torch.manual_seed(0)
    model = _WeightTiedModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_custom_autograd_function() -> None:
    """Parameter-grad comparison covers custom autograd.Function nodes."""

    torch.manual_seed(0)
    model = _CustomParamModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_hygiene_data_parallel() -> None:
    """Backward validation unwraps DataParallel wrappers."""

    model = nn.DataParallel(_TinyBackwardModel())
    model.module.cpu()
    x = torch.randn(2, 3, requires_grad=True)
    assert backward_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_hygiene_opaque_wrapper() -> None:
    """Backward validation rejects opaque TorchScript wrappers."""

    model = torch.jit.trace(_TinyBackwardModel(), torch.randn(2, 3))
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="torch.jit"):
        backward_validation.validate_backward_pass(model, x, random_seed=42)


def test_accumulategrad_labels_deterministic_across_captures() -> None:
    """AccumulateGrad labels are stable across seeded captures."""

    torch.manual_seed(0)
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace1 = tl.trace(model, x, capture=CaptureOptions(save_grads="all", random_seed=42))
    trace1.log_backward(_output_loss(trace1))
    labels1 = {
        grad_fn_handle.label
        for grad_fn_handle in trace1.grad_fn_logs.values()
        if grad_fn_handle.type == "accumulategrad"
    }
    trace1.cleanup()

    trace2 = tl.trace(model, x, capture=CaptureOptions(save_grads="all", random_seed=42))
    trace2.log_backward(_output_loss(trace2))
    labels2 = {
        grad_fn_handle.label
        for grad_fn_handle in trace2.grad_fn_logs.values()
        if grad_fn_handle.type == "accumulategrad"
    }
    trace2.cleanup()

    assert labels1 == labels2


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:`layers_to_save` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:`random_seed` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:`save_grads` is deprecated:DeprecationWarning")
def test_validate_backward_pass_perturbed() -> None:
    """The inert saved-grad perturbation option is loudly unsupported."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.warns(DeprecationWarning, match="perturb_saved_grads"):
        with pytest.raises(ValueError, match="unsupported"):
            tl_validation.validate_backward_pass(model, x, perturb_saved_grads=True)


@pytest.mark.smoke
def test_peak_memory_tracking_populated() -> None:
    """Trace stores flat backward peak-memory tracking metadata."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.has_backward_pass
    assert isinstance(trace.backward_peak_memory, int)
    assert trace.backward_memory_backend in {"cpu", "cuda", "mps"}


@pytest.mark.smoke
def test_higher_order_grads_basic_support() -> None:
    """create_graph=True backward calls run through capture."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace), create_graph=True)
    assert trace.num_backward_passes == 1


# ---------------------------------------------------------------------------
# Dual-review fix round: param-grad event authority, event immutability,
# exact bracketing, restored-trace streams, and cleanup disarm.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_param_grad_records_rebuild_from_event_spine() -> None:
    """A forced scratch rebuild reconstructs Param._grad_records from events."""
    from torchlens.backends.torch import backward as backward_mod

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        loss.backward()

    before = {
        address: [
            (record.ordinal, record.backward_pass_index, record.shape, record.memory)
            for record in param_log._grad_records
        ]
        for address, param_log in trace.param_logs.items()
    }
    payloads_before = {
        address: [record.grad for record in param_log._grad_records]
        for address, param_log in trace.param_logs.items()
    }
    assert any(before.values()), "expected captured param gradient records"
    assert all(len(records) == 2 for records in before.values() if records)

    for param_log in trace.param_logs.values():
        param_log._grad_records = []
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)

    after = {
        address: [
            (record.ordinal, record.backward_pass_index, record.shape, record.memory)
            for record in param_log._grad_records
        ]
        for address, param_log in trace.param_logs.items()
    }
    assert after == before
    for address, param_log in trace.param_logs.items():
        for record, payload in zip(param_log._grad_records, payloads_before[address]):
            assert record.grad is payload, "rebuild must reuse the event-held payload"


@pytest.mark.smoke
def test_param_grad_incremental_fold_matches_scratch_rebuild() -> None:
    """The param-inclusive snapshot proves fold == scratch across passes."""
    from torchlens.backends.torch import backward as backward_mod

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        loss.backward()

    incremental_snapshot = _backward_projection_snapshot(trace)
    assert incremental_snapshot["param_grads"], "oracle must include parameter state"
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)
    assert _backward_projection_snapshot(trace) == incremental_snapshot


@pytest.mark.smoke
def test_param_grad_reconciliation_counts_multiplicity() -> None:
    """Duplicating one projected param record now fails reconciliation."""
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    _invariant_check(trace)  # positive control

    param_log = next(
        param_log for param_log in trace.param_logs.values() if param_log._grad_records
    )
    duplicated = param_log._grad_records[0]
    param_log._grad_records.append(duplicated)
    with pytest.raises(MetadataInvariantError, match="by multiplicity"):
        _invariant_check(trace)
    param_log._grad_records.pop()
    _invariant_check(trace)


@pytest.mark.smoke
def test_op_grad_reconciliation_counts_multiplicity() -> None:
    """Duplicating one projected op record now fails reconciliation."""
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    _invariant_check(trace)  # positive control

    victim = next(op for op in trace.layer_list if op._slot("_grad_records"))
    records = victim._slot("_grad_records")
    records.append(records[0])
    with pytest.raises(MetadataInvariantError, match="by multiplicity"):
        _invariant_check(trace)
    records.pop()
    _invariant_check(trace)


@pytest.mark.smoke
def test_grad_fn_discovered_source_is_frozen_by_the_writer() -> None:
    """In-place mutation of GradFnDiscovered.source raises instead of biting."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import GradFnDiscovered

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    events = _ensure_backward_event_stream(trace).backward_events
    discovered = next(event for event in events if isinstance(event, GradFnDiscovered))
    with pytest.raises(TypeError):
        discovered.source["class_source_file"] = "/tmp/planted-mutation.py"  # type: ignore[index]
    assert dict(discovered.source) is not discovered.source  # copies still work


@pytest.mark.smoke
def test_higher_order_discovery_bracketing_is_armed() -> None:
    """A created_in_pass discovery moved past its pass end fails the invariant."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd, GradFnDiscovered
    from torchlens.validation.invariants import MetadataInvariantError

    class _HigherOrderModel(nn.Module):
        """Tiny nonlinear model with differentiable first gradients."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.tanh(x) ** 3).sum()

    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    trace = tl.trace(_HigherOrderModel(), x, save_grads="all")
    loss = trace[trace.output_layers[0]].out
    first_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
    torch.autograd.grad(first_grad.sum(), x, retain_graph=True)
    events = _ensure_backward_event_stream(trace).backward_events
    _invariant_check(trace)  # positive control

    created = [
        event
        for event in events
        if isinstance(event, GradFnDiscovered) and event.created_in_pass is not None
    ]
    assert created, "create_graph autograd.grad must discover higher-order grad-fns"
    victim = created[0]
    end = next(
        event
        for event in events
        if isinstance(event, BackwardPassEnd) and event.pass_index == victim.created_in_pass
    )
    # Move ONLY the discovery after its pass end and renumber every event
    # monotonically in list order (the sol probe): all other pass-scoped
    # events stay inside their brackets, so a failure here proves the new
    # created_in_pass bracket check specifically is armed.
    original_order = list(events)
    original_seqs = [event.seq for event in events]
    events.remove(victim)
    events.insert(events.index(end) + 1, victim)
    for renumbered_seq, event in enumerate(events, start=1):
        object.__setattr__(event, "seq", renumbered_seq)
    with pytest.raises(MetadataInvariantError, match="higher-order GradFnDiscovered"):
        _invariant_check(trace)
    events[:] = original_order
    for original_seq, event in zip(original_seqs, events):
        object.__setattr__(event, "seq", original_seq)
    _invariant_check(trace)


@pytest.mark.smoke
def test_pass_brackets_reject_partial_interleaving() -> None:
    """Two pass brackets that partially overlap fail the invariant."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        loss.backward()
    events = _ensure_backward_event_stream(trace).backward_events
    _invariant_check(trace)  # positive control

    end_one = next(
        event for event in events if isinstance(event, BackwardPassEnd) and event.pass_index == 1
    )
    start_two = next(
        event
        for event in events
        if isinstance(event, BackwardPassStart) and event.pass_index == 2
    )
    # Interleave: end(1) slides just after start(2) and every seq is
    # renumbered monotonically -> [1 .. [2 .. 1] .. 2]. Each pass-scoped fact
    # still sits inside its own bracket, so only the new partial-overlap
    # check can fire.
    original_order = list(events)
    original_seqs = [event.seq for event in events]
    events.remove(end_one)
    events.insert(events.index(start_two) + 1, end_one)
    for renumbered_seq, event in enumerate(events, start=1):
        object.__setattr__(event, "seq", renumbered_seq)
    with pytest.raises(MetadataInvariantError, match="partially overlaps"):
        _invariant_check(trace)
    events[:] = original_order
    for original_seq, event in zip(original_seqs, events):
        object.__setattr__(event, "seq", original_seq)
    _invariant_check(trace)


@pytest.mark.smoke
def test_restored_trace_supports_backward_capture() -> None:
    """A pickled-and-restored trace records a fresh backward correctly."""
    import pickle

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
        save_mode="reference",
    )
    loss = _output_loss(trace)
    restored = pickle.loads(pickle.dumps(trace))
    assert len(restored._capture_events.backward_events) == 0
    assert "_backward_projection_revision" not in restored.__dict__
    assert "_backward_projection_fold_state" not in restored.__dict__

    restored.log_backward(loss)
    assert restored.num_backward_passes == 1
    assert restored.grad_fn_logs
    assert restored.backward_pass_logs[1].status == "ok"
    _invariant_check(restored)


@pytest.mark.smoke
def test_double_restore_replaces_stale_stream() -> None:
    """__setstate__ replaces a reused object's stream based on incoming state."""
    import pickle

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace._capture_events.backward_events

    pickled_state = pickle.dumps(trace)
    restored = pickle.loads(pickled_state)
    # Plant an event on the fresh stream, then restore AGAIN onto the same
    # object: the stale stream (and its planted event) must not survive.
    restored._capture_events.backward_events.append("SENTINEL")
    restored.__setstate__(trace.__getstate__())
    assert len(restored._capture_events.backward_events) == 0

    # And the stream never leaks into later pickle state.
    assert "_capture_events" not in restored.__getstate__()
    assert "_capture_events" not in trace.__getstate__()
    assert "_backward_projection_revision" not in trace.__getstate__()
    assert "_backward_projection_fold_state" not in trace.__getstate__()


@pytest.mark.smoke
def test_cleanup_disarms_backward_triggers() -> None:
    """A user backward after cleanup() must not raise from lingering hooks."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    out = trace[trace.output_layers[0]].out
    trace.log_backward(out.sum(), retain_graph=True)
    trace.cleanup()

    # The tensor hooks and grad_fn hooks registered on the user's still-live
    # graph fire during this backward; disarmed, they must silently no-op.
    out.sum().backward()
    assert x.grad is not None
