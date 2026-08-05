"""Round-22 conditional/taken-branch hardening regressions.

Adversarial audit findings (round22 condbranch): order-dependent branch
erasure, 1:N predicate reuse, same-line nested-ternary cross-wiring,
decorated-forward scope loss, short-circuited elif evaluation claims,
``bool_value_at_run`` vs ``fired`` contradictions, and invisible non-bool
tensor truthiness. Every test asserts the recorded conditional structure
against the ACTUALLY executed branches, and that no false-``fired`` arm is
introduced (an arm never claims execution that did not happen).

All models live at module level so the file stays AST-readable for the
conditional classifier, and every branch arm body sits on its own source
line so degraded line-only attribution works on Python 3.10 (no column
info before 3.11).
"""

from __future__ import annotations

import functools
import warnings
from typing import Callable, Optional

import pytest
import torch
import torch.nn as nn

from torchlens import trace as trace_fn
from torchlens.data_classes.op import Op
from torchlens.data_classes.trace import Trace

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _log_model(model: nn.Module, x: torch.Tensor) -> Trace:
    """Log one forward pass of a small inline test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.

    Returns
    -------
    Trace
        Postprocessed model log.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trace_fn(model, x)


def _find_only_layer(trace: Trace, func_name: str) -> Op:
    """Return the single non-output layer whose ``func_name`` matches.

    Parameters
    ----------
    trace:
        Trace to search.
    func_name:
        Captured function name to match.

    Returns
    -------
    Op
        The unique matching layer.
    """

    matches = [
        layer for layer in trace.layer_list if layer.func_name == func_name and not layer.is_output
    ]
    assert len(matches) == 1, f"expected one {func_name!r} layer, found {len(matches)}"
    return matches[0]


def _layer_names(trace: Trace) -> set[str]:
    """Return the set of non-output captured function names.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    set[str]
        Captured ``func_name`` values.
    """

    return {layer.func_name for layer in trace.layer_list if not layer.is_output}


def _assert_no_false_fired(trace: Trace, executed: set[str], not_executed: set[str]) -> None:
    """Assert no arm claims execution of a branch that did not run.

    Parameters
    ----------
    trace:
        Trace whose public conditionals are checked.
    executed:
        Function names of ops that genuinely executed this forward.
    not_executed:
        Function names of ops on arms that did NOT run; they must be absent
        from the capture and no fired arm may reference them.
    """

    captured = _layer_names(trace)
    for name in not_executed:
        assert name not in captured, f"non-executed op {name!r} appears in the capture"
    for conditional in trace.conditionals:
        for arm in conditional.arms:
            if not arm.fired:
                continue
            # A fired arm must be backed by at least one actually-executed op.
            fired_ops = set(arm.execution_ops)
            assert fired_ops, "fired arm carries no execution ops"
            fired_funcs = {
                trace.layer_dict_all_keys[label].func_name
                for label in fired_ops
                if label in trace.layer_dict_all_keys
            }
            assert fired_funcs <= (executed | {"none"}), (
                f"fired arm references non-executed ops: {fired_funcs - executed}"
            )


def _assert_bool_value_never_contradicts_fired(trace: Trace) -> None:
    """Assert ``bool_value_at_run`` is consistent with the fired arm.

    For single-pass captures: when a conditional fired its then arm, a
    non-``None`` then-arm value must be ``True``; when it fired any other
    arm, a non-``None`` then-arm value must be ``False``. ``None`` (honest
    unknown, e.g. compound tests) never contradicts.

    Parameters
    ----------
    trace:
        Trace whose public conditionals are checked.
    """

    for conditional in trace.conditionals:
        for arm in conditional.arms:
            if arm.kind != "then" or arm.bool_value_at_run is None:
                continue
            if conditional.fired_arm_kind == "then":
                assert arm.bool_value_at_run is True, (
                    f"then arm fired but bool_value_at_run={arm.bool_value_at_run}"
                )
            elif conditional.fired_arm_kind is not None:
                assert arm.bool_value_at_run is False, (
                    f"{conditional.fired_arm_kind} fired but then arm records "
                    f"bool_value_at_run={arm.bool_value_at_run}"
                )


# ---------------------------------------------------------------------------
# F1: order-dependence / branch erasure (assert or while consuming first)
# ---------------------------------------------------------------------------


class AssertThenIfModel(nn.Module):
    """Predicate consumed by ``assert`` BEFORE the ``if`` on the same tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the guarded conditional forward pass."""
        ok = (x > 0).all()
        assert ok, "sanity guard"
        if ok:
            x = torch.relu(x)
        return x * 2


class IfOnlyModel(nn.Module):
    """Control: the identical conditional without the preceding ``assert``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the plain conditional forward pass."""
        ok = (x > 0).all()
        if ok:
            x = torch.relu(x)
        return x * 2


class WhileThenIfModel(nn.Module):
    """Predicate consumed by a ``while`` line first, then by an ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the while-guarded conditional forward pass."""
        big = x.sum() > 1000
        while big:
            x = x * 0.5
            break
        if big:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


def test_assert_then_if_still_materializes_branch() -> None:
    """A prior assert consumption must not erase the later if conditional."""

    guarded = _log_model(AssertThenIfModel(), torch.ones(2, 2))
    control = _log_model(IfOnlyModel(), torch.ones(2, 2))

    assert len(guarded.conditional_records) == 1
    assert len(control.conditional_records) == 1

    guarded_relu = _find_only_layer(guarded, "relu")
    control_relu = _find_only_layer(control, "relu")
    assert guarded_relu.conditional_branch_stack == [(0, "then")]
    assert guarded_relu.conditional_branch_stack == control_relu.conditional_branch_stack

    (conditional,) = list(guarded.conditionals)
    assert conditional.fired_arm_kind == "then"
    bool_layer = _find_only_layer(guarded, "all")
    assert bool_layer.conditional_context_kind == "if_test"
    assert bool_layer.is_terminal_conditional_bool is True
    assert bool_layer.terminal_conditional_id == 0

    _assert_no_false_fired(guarded, {"relu", "all", "__gt__", "mul"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(guarded)


def test_while_then_if_still_materializes_branch() -> None:
    """A prior while consumption must not erase the later if/else conditional."""

    trace = _log_model(WhileThenIfModel(), torch.ones(2, 2))

    assert len(trace.conditional_records) == 1
    sigmoid_layer = _find_only_layer(trace, "sigmoid")
    assert sigmoid_layer.conditional_branch_stack == [(0, "else")]
    (conditional,) = list(trace.conditionals)
    assert conditional.fired_arm_kind == "else"

    _assert_no_false_fired(trace, {"sigmoid", "sum", "__gt__", "mul"}, {"relu"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F2: predicate reuse — one bool gating two separate if statements
# ---------------------------------------------------------------------------


class PredicateReuseModel(nn.Module):
    """The same bool tensor gates TWO separate ``if`` statements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both gated branches."""
        ok = (x > 0).all()
        if ok:
            x = torch.relu(x)
        x = x * 2
        if ok:
            x = torch.sigmoid(x)
        return x + 1


def test_predicate_reuse_materializes_every_branch_consumer() -> None:
    """Both ifs gated by one bool are recorded, each with its own arm roles."""

    trace = _log_model(PredicateReuseModel(), torch.ones(2, 2))

    assert len(trace.conditional_records) == 2
    events = sorted(trace.conditional_records, key=lambda event: event.id)
    bool_layer = _find_only_layer(trace, "all")
    for event in events:
        assert event.bool_layers == [bool_layer.layer_label]

    relu_layer = _find_only_layer(trace, "relu")
    sigmoid_layer = _find_only_layer(trace, "sigmoid")
    assert relu_layer.conditional_branch_stack == [(events[0].id, "then")]
    assert sigmoid_layer.conditional_branch_stack == [(events[1].id, "then")]
    assert {conditional.fired_arm_kind for conditional in trace.conditionals} == {"then"}

    # The shared bool keeps a deterministic primary conditional id.
    assert bool_layer.terminal_conditional_id == events[0].id
    assert bool_layer.is_terminal_conditional_bool is True

    _assert_no_false_fired(trace, {"relu", "sigmoid", "all", "__gt__", "mul", "add"}, set())
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F3: same-line nested ternary must not cross-wire bools across conditionals
# ---------------------------------------------------------------------------


class SameLineNestedTernaryModel(nn.Module):
    """Two ternary tests on ONE source line (nested ternary)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the one-line nested ternary."""
        y = (torch.relu(x) if (x > 0).all() else torch.sigmoid(x)) if (x < 10).all() else torch.tanh(x)  # fmt: skip  # noqa: E501
        return y * 2


class MultiLineNestedTernaryModel(nn.Module):
    """Formatter-wrapped nested ternary: outer test on its OWN line."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the multi-line nested ternary."""
        y = (
            (torch.relu(x) if (x > 0).all() else torch.sigmoid(x))
            if (x < 10).all()
            else torch.tanh(x)
        )
        return y * 2


def test_same_line_nested_ternary_does_not_cross_wire_bools() -> None:
    """Either both conditionals materialize with their OWN bools, or the
    classification fails closed — never one event owning a foreign bool."""

    trace = _log_model(SameLineNestedTernaryModel(), torch.ones(2, 2))

    events = trace.conditional_records
    # The audited failure mode was exactly ONE event (the inner ternary)
    # carrying BOTH bools, with the outer bool promoted to the inner event's
    # terminal bool. Honest outcomes: fail-closed (no events, no conditional
    # claims) on line-only runtimes, or two events each owning its own bool
    # where column info permits precise classification.
    assert len(events) in (0, 2), f"cross-wire signature: {len(events)} event(s)"
    if len(events) == 2:
        bool_lists = [event.bool_layers for event in events]
        assert all(len(bool_list) == 1 for bool_list in bool_lists)
        assert bool_lists[0] != bool_lists[1]
    else:
        for layer in trace.layer_list:
            assert layer.conditional_branch_stack == []
            assert getattr(layer, "is_terminal_conditional_bool", False) is False

    # relu executed; tanh/sigmoid did not — no arm may claim they fired.
    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "__lt__", "mul"}, {"sigmoid", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_multi_line_nested_ternary_does_not_cross_wire_bools() -> None:
    """A formatter-wrapped nested ternary must not absorb the outer test's
    bool into the inner conditional when the runtime misattributes the
    consumption line (py3.10 reports the inner ternary's line for both)."""

    trace = _log_model(MultiLineNestedTernaryModel(), torch.ones(2, 2))

    bool_layers = [
        layer
        for layer in trace.layer_list
        if getattr(layer, "is_scalar_bool", False) and not layer.is_output
    ]
    assert len(bool_layers) == 2
    creation_lines = {
        layer.layer_label: next(
            frame.line_number for frame in layer.code_context if frame.file == __file__
        )
        for layer in bool_layers
    }

    events = trace.conditional_records
    assert len(events) in (1, 2), f"cross-wire signature: {len(events)} event(s)"
    for event in events:
        # Every bool attached to an event must have been CREATED on the
        # event's own test line (both tests here are inline single-line
        # expressions) — a foreign bool on the record is the cross-wire.
        test_line = event.test_span[0]
        for bool_label in event.bool_layers:
            assert creation_lines[bool_label] == test_line, (
                f"event at test line {test_line} owns foreign bool "
                f"{bool_label} created at line {creation_lines[bool_label]}"
            )

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "__lt__", "mul"}, {"sigmoid", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F4: decorated forward — stacked and multi-line decorators
# ---------------------------------------------------------------------------


def _passthrough(func: Callable) -> Callable:
    """Return a wrapping passthrough decorator preserving metadata."""

    @functools.wraps(func)
    def inner(*args: object, **kwargs: object) -> object:
        """Delegate to the wrapped callable."""
        return func(*args, **kwargs)

    return inner


def _configurable_passthrough(
    label: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Return a decorator factory mimicking HF-style multi-line decorators."""

    def decorate(func: Callable) -> Callable:
        """Wrap the callable with metadata preserved."""

        @functools.wraps(func)
        def inner(*args: object, **kwargs: object) -> object:
            """Delegate to the wrapped callable."""
            return func(*args, **kwargs)

        return inner

    return decorate


class DoublyDecoratedForwardModel(nn.Module):
    """``forward`` carries TWO stacked decorators."""

    @_passthrough
    @_passthrough
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decorated conditional forward pass."""
        if (x > 0).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


class MultiLineDecoratedForwardModel(nn.Module):
    """``forward`` carries one decorator call spanning several lines."""

    @_configurable_passthrough(
        label="documented",
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decorated conditional forward pass."""
        if (x > 0).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


@pytest.mark.parametrize(
    "model_factory",
    [DoublyDecoratedForwardModel, MultiLineDecoratedForwardModel],
    ids=["two-stacked-decorators", "multi-line-decorator"],
)
def test_multi_decorator_forward_attributes_taken_branch(
    model_factory: Callable[[], nn.Module],
) -> None:
    """Decorated forwards attribute the taken branch exactly, not degraded."""

    trace = _log_model(model_factory(), torch.ones(2, 2))

    assert len(trace.conditional_records) == 1
    relu_layer = _find_only_layer(trace, "relu")
    assert relu_layer.conditional_branch_stack == [(0, "then")]
    (conditional,) = list(trace.conditionals)
    assert conditional.fired_arm_kind == "then"
    then_arm = conditional.arms[0]
    assert then_arm.fired is True
    assert then_arm.condition_evaluated is True

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F5: short-circuited elif tests must not claim condition_evaluated
# ---------------------------------------------------------------------------


class ElifShortCircuitModel(nn.Module):
    """A True if-test short-circuits the elif test entirely."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the elif ladder."""
        if (x > 0).all():
            y = torch.relu(x)
        elif (x < -100).all():
            y = torch.sigmoid(x)
        else:
            y = torch.tanh(x)
        return y * 2


def test_elif_condition_evaluated_false_when_short_circuited() -> None:
    """An elif test that never ran reports condition_evaluated=False."""

    trace = _log_model(ElifShortCircuitModel(), torch.ones(2, 2))

    (conditional,) = list(trace.conditionals)
    then_arm, elif_arm, else_arm = conditional.arms
    assert (then_arm.kind, elif_arm.kind, else_arm.kind) == ("then", "elif", "else")

    assert then_arm.fired is True
    assert then_arm.condition_evaluated is True
    assert then_arm.bool_value_at_run is True

    # The elif test NEVER executed: no evaluation claim, no evaluation edge.
    assert elif_arm.fired is False
    assert elif_arm.condition_evaluated is False
    assert elif_arm.bool_value_at_run is None
    assert elif_arm.evaluation_entry_edge is None
    assert elif_arm.evaluation_ops == []

    assert else_arm.fired is False

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul"}, {"sigmoid", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_elif_condition_evaluated_true_when_reached() -> None:
    """Converse control: a genuinely evaluated elif test reports True."""

    trace = _log_model(ElifShortCircuitModel(), torch.full((2, 2), -200.0))

    (conditional,) = list(trace.conditionals)
    then_arm, elif_arm, _else_arm = conditional.arms
    assert then_arm.fired is False
    assert then_arm.condition_evaluated is True
    assert then_arm.bool_value_at_run is False
    assert elif_arm.fired is True
    assert elif_arm.condition_evaluated is True

    _assert_no_false_fired(trace, {"sigmoid", "all", "__gt__", "__lt__", "mul"}, {"relu", "tanh"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F6: bool_value_at_run must never contradict fired (negation, short-circuit)
# ---------------------------------------------------------------------------


class NegatedIfModel(nn.Module):
    """``if not pred:`` — then fires exactly when the raw bool is False."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the negated conditional."""
        if not (x > 0).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


class ShortCircuitAndModel(nn.Module):
    """``if b1 and b2:`` with b1 True, b2 False — else fires."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the compound conditional."""
        if (x > 0).all() and (x > 100).all():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


def test_negated_if_bool_value_reflects_test_outcome() -> None:
    """Under ``if not pred:`` the recorded value is the TEST outcome."""

    trace = _log_model(NegatedIfModel(), -torch.ones(2, 2))

    (conditional,) = list(trace.conditionals)
    assert conditional.fired_arm_kind == "then"
    then_arm = conditional.arms[0]
    assert then_arm.fired is True
    # Raw bool is False; the test outcome (not False) is True and must not
    # read as "condition False yet then fired".
    assert then_arm.bool_value_at_run is True

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)


def test_short_circuit_and_never_reports_true_for_unfired_then() -> None:
    """A compound and-test refuses a single-bool value instead of lying."""

    trace = _log_model(ShortCircuitAndModel(), torch.ones(2, 2))

    (conditional,) = list(trace.conditionals)
    assert conditional.fired_arm_kind == "else"
    then_arm = conditional.arms[0]
    assert then_arm.fired is False
    # b1 is True but the test outcome is False: promoting b1's raw value
    # would contradict fired. Honest answer for compound tests is None.
    assert then_arm.bool_value_at_run is None
    assert then_arm.condition_evaluated is True

    _assert_no_false_fired(trace, {"sigmoid", "all", "__gt__", "mul"}, {"relu"})
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F7 (documented gap, DEFERRED): non-bool tensor truthiness stays invisible
# ---------------------------------------------------------------------------


class FloatTruthinessModel(nn.Module):
    """``if x.sum():`` — ``__bool__`` on a NON-bool scalar tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the truthiness-gated conditional."""
        if x.sum():
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y * 2


@pytest.mark.parametrize(
    ("input_tensor", "executed", "not_executed"),
    [
        (torch.ones(2, 2), {"relu", "sum", "mul"}, {"sigmoid"}),
        (torch.zeros(2, 2), {"sigmoid", "sum", "mul"}, {"relu"}),
    ],
    ids=["truthy-then", "falsy-else"],
)
def test_float_truthiness_stays_documented_false_negative(
    input_tensor: torch.Tensor,
    executed: set[str],
    not_executed: set[str],
) -> None:
    """DELIBERATE false negative, pinned: non-bool tensor truthiness is a real
    bool consumption but is NOT recorded as a conditional. Recording it would
    materialize arm edges whose predicate the runnable witness-obligation
    registry cannot witness (only ``is_scalar_bool`` ops receive predicate
    witnesses), so every level="runnable" save of such a model would refuse at
    producer preflight (caught by the round-22 smoke gate on the raw-input
    truthiness escape-witness tests). Recording is deferred until the runnable
    contract gains a truthiness predicate witness family. This pin flips to a
    positive capture test at that point; meanwhile nothing may false-fire."""

    trace = _log_model(FloatTruthinessModel(), input_tensor)

    assert trace.conditional_records == []
    assert list(trace.conditionals) == []
    sum_layer = _find_only_layer(trace, "sum")
    assert sum_layer.is_terminal_conditional_bool is not True
    for layer in trace.layer_list:
        assert layer.conditional_branch_stack == []

    _assert_no_false_fired(trace, executed, not_executed)
    _assert_bool_value_never_contradicts_fired(trace)


# ---------------------------------------------------------------------------
# F8 (documented gap): comprehension-embedded ternary never false-fires
# ---------------------------------------------------------------------------


class ComprehensionTernaryModel(nn.Module):
    """Data-dependent ternary inside a list comprehension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the per-element comprehension ternary."""
        parts = [torch.relu(c) if (c > 0).all() else torch.sigmoid(c) for c in x.unbind(0)]
        return torch.stack(parts) * 2


def test_comprehension_ternary_never_false_fires() -> None:
    """Documented gap: comprehension-scope arm attribution may fail closed,
    but the record must never claim a branch that did not execute."""

    trace = _log_model(ComprehensionTernaryModel(), torch.ones(3, 2))

    # relu ran per element; sigmoid never did. Whether or not the runtime can
    # attribute arms inside a <listcomp> scope (it cannot before Python 3.12
    # comprehension inlining), a wrong-arm or else-fired claim is forbidden.
    for conditional in trace.conditionals:
        assert conditional.fired_arm_kind in (None, "then")
        for arm in conditional.arms:
            if arm.kind == "else":
                assert arm.fired is False

    _assert_no_false_fired(trace, {"relu", "all", "__gt__", "mul", "unbind", "stack"}, {"sigmoid"})
    _assert_bool_value_never_contradicts_fired(trace)
