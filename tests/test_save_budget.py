"""Save-budget refusal: bound retained bytes honestly instead of OOM-killing.

``tl.trace(model, x)`` retains every operation's output by default, which at
frontier shapes means a first-time user gets an OOM kill or an allocator error
from deep inside torch rather than an explanation. These tests pin the honest
replacement: a per-device running ceiling on retained payload bytes that stops
capture with :class:`SaveBudgetExceededError`, naming the committed footprint,
the tripping operation, and the remedies.

Two properties matter as much as the refusal itself and are tested explicitly:

* The accepted path is unchanged. The default ``"auto"`` budget (half of
  available memory) must never fire on an ordinary model, and a capture that
  retains nothing in RAM -- disk-streamed or metadata-only -- must not be charged
  at all, because it was never going to OOM.
* The reported figure is honest. It is a *lower bound* on the finished capture's
  footprint (the forward was still running when it tripped), and the message says
  so rather than extrapolating a total it cannot know.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._save_budget import (
    DEFAULT_SAVE_BUDGET_FRACTION,
    SaveBudget,
    SaveBudgetExceededError,
    available_device_bytes,
    format_bytes,
    resolve_save_budget,
)
from torchlens.data_classes.trace import Trace
from torchlens.options import CaptureOptions


def _model() -> nn.Module:
    """Return a small model whose activations are a couple of KB each.

    Returns
    -------
    nn.Module
        Two-linear-layer model.
    """

    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))


def _input() -> torch.Tensor:
    """Return the matching input batch.

    Returns
    -------
    torch.Tensor
        Batch of 8 vectors of width 64 (2 KB per float32 activation).
    """

    return torch.randn(8, 64)


# ---------------------------------------------------------------------------
# Option resolution: invalid budgets fail loudly, never silently disable
# ---------------------------------------------------------------------------


def test_auto_is_the_default_and_resolves_to_a_fraction() -> None:
    """``"auto"`` is the shipped default and means a headroom fraction."""

    assert CaptureOptions().save_budget == "auto"
    spec = resolve_save_budget("auto")
    assert spec is not None
    assert spec.fraction == DEFAULT_SAVE_BUDGET_FRACTION
    assert spec.absolute_bytes is None
    assert "default" in spec.source


def test_none_disables_budgeting() -> None:
    """``None`` is the documented off switch."""

    assert resolve_save_budget(None) is None
    assert SaveBudget.from_option(None) is None


def test_float_resolves_to_a_fraction_and_int_to_absolute_bytes() -> None:
    """Both documented numeric spellings resolve distinctly."""

    fraction_spec = resolve_save_budget(0.25)
    assert fraction_spec is not None
    assert fraction_spec.fraction == 0.25
    assert fraction_spec.absolute_bytes is None

    absolute_spec = resolve_save_budget(4096)
    assert absolute_spec is not None
    assert absolute_spec.absolute_bytes == 4096
    assert absolute_spec.fraction is None
    assert "4.00 KB" in absolute_spec.source


@pytest.mark.parametrize(
    "value",
    [0.0, -0.5, 1.5, 0, -1, True, False, "half", "AUTO", object()],
)
def test_invalid_budget_raises_rather_than_silently_disabling(value: object) -> None:
    """A malformed budget must fail loudly; silently unguarding is the bug."""

    with pytest.raises(ValueError):
        resolve_save_budget(value)  # type: ignore[arg-type]


def test_format_bytes_is_readable_at_every_scale() -> None:
    """Byte rendering stays readable from bytes to terabytes."""

    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(3 * 1024**3) == "3.00 GB"
    assert format_bytes(2 * 1024**4) == "2.00 TB"


def test_unmeasurable_device_is_unbudgeted_not_assumed_infinite() -> None:
    """Meta tensors have no storage, so headroom is reported unmeasurable."""

    assert available_device_bytes(torch.device("meta")) is None
    budget = SaveBudget.from_option("auto")
    assert budget is not None
    budget.charge("op", torch.device("meta"), 1)
    assert budget.unbudgeted_devices() == ("meta",)
    assert budget.tripped is False


def test_host_headroom_is_measurable_on_this_platform() -> None:
    """The CPU budget is only meaningful if host headroom can be read."""

    available = available_device_bytes(torch.device("cpu"))
    assert available is None or available > 0


# ---------------------------------------------------------------------------
# The accepted path stays exactly as it was
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_default_budget_does_not_fire_on_an_ordinary_model() -> None:
    """The shipped default must never refuse a normal capture."""

    trace = tl.trace(_model(), _input())
    assert trace.num_saved_ops > 0
    assert int(trace.saved_activation_memory) > 0


@pytest.mark.smoke
def test_default_capture_is_unchanged_by_the_budget() -> None:
    """A default capture and an unbudgeted capture agree on what was saved."""

    model = _model()
    x = _input()
    budgeted = tl.trace(model, x)
    unbudgeted = tl.trace(model, x, capture=CaptureOptions(save_budget=None))

    assert budgeted.layer_labels == unbudgeted.layer_labels
    assert budgeted.num_saved_ops == unbudgeted.num_saved_ops
    assert int(budgeted.saved_activation_memory) == int(unbudgeted.saved_activation_memory)


def test_disabled_budget_allows_what_a_tiny_budget_refuses() -> None:
    """The same capture succeeds with ``None`` and refuses with a tiny cap."""

    model = _model()
    x = _input()
    assert tl.trace(model, x, capture=CaptureOptions(save_budget=None)).num_saved_ops > 0
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(model, x, capture=CaptureOptions(save_budget=64))


def test_metadata_only_capture_is_never_charged() -> None:
    """``layers_to_save="none"`` retains no payloads, so it cannot trip.

    The graph is still captured; refusing here would refuse a capture that was
    never going to allocate anything.
    """

    trace = tl.trace(
        _model(), _input(), layers_to_save="none", capture=CaptureOptions(save_budget=64)
    )
    assert len(trace.layer_labels) > 0
    assert trace.num_saved_ops == 0


def test_disk_streamed_payloads_are_not_charged(tmp_path: Path) -> None:
    """Payloads streamed to disk cost no process memory, so they are not charged."""

    trace = tl.trace(
        _model(),
        _input(),
        save=tl.func("relu"),
        storage=tl.to_disk(str(tmp_path / "run.tlspec")),
        capture=CaptureOptions(save_budget=64),
    )
    assert len(trace.layer_labels) > 0


# ---------------------------------------------------------------------------
# The refusal itself
# ---------------------------------------------------------------------------


def test_absolute_budget_refuses_with_structured_fields() -> None:
    """The refusal carries numbers, so callers never parse the message."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1024))

    fields = excinfo.value.fields
    assert fields["budget_bytes"] == 1024
    assert fields["committed_bytes"] > 1024
    assert fields["device"] == "cpu"
    assert fields["num_saved"] >= 1
    assert isinstance(fields["label"], str) and fields["label"]


def test_refusal_message_names_footprint_site_and_remedies() -> None:
    """The message must explain, not just fail."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1024))
    message = str(excinfo.value)

    assert "save budget" in message
    assert "committed so far" in message
    assert "1.00 KB" in message, "the configured budget must be quoted"
    assert "tripped while saving" in message
    # Honesty: the figure is a lower bound and the message says so, rather than
    # extrapolating a total from an incomplete forward.
    assert "LOWER BOUND" in message
    assert "save=tl.func" in message
    assert "storage=tl.to_disk" in message
    assert "save_budget=None" in message


def test_refusal_names_the_configured_source_not_a_bare_number() -> None:
    """A user who set a fraction sees their fraction quoted back."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1e-9))
    assert "save_budget=1e-09" in str(excinfo.value)


def test_predicate_save_path_is_also_budgeted() -> None:
    """Both activation-save paths charge the budget, not just the default one."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(
            _model(),
            _input(),
            save=tl.func("relu"),
            capture=CaptureOptions(save_budget=64),
        )
    assert "relu" in excinfo.value.fields["label"]


def test_budget_refusal_leaves_the_model_reusable() -> None:
    """A refused capture must not leak capture state onto the model."""

    model = _model()
    x = _input()
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(model, x, capture=CaptureOptions(save_budget=64))

    # The model still runs normally, and a subsequent ordinary capture works.
    assert model(x).shape == (8, 64)
    trace = tl.trace(model, x)
    assert trace.num_saved_ops > 0


def test_typed_error_is_reachable_from_public_errors_namespace() -> None:
    """The refusal type is part of the public error surface."""

    import torchlens.errors as errors

    assert errors.SaveBudgetExceededError is SaveBudgetExceededError
    assert "SaveBudgetExceededError" in errors.__all__
    assert issubclass(SaveBudgetExceededError, errors.CaptureError)


# ---------------------------------------------------------------------------
# Accountant unit behavior
# ---------------------------------------------------------------------------


def test_accountant_charges_cumulatively_and_trips_once_over() -> None:
    """The ledger accumulates across ops and trips on the crossing charge."""

    budget = SaveBudget.from_option(100)
    assert budget is not None
    cpu = torch.device("cpu")
    budget.charge("a", cpu, 40)
    budget.charge("b", cpu, 40)
    assert budget.tripped is False
    with pytest.raises(SaveBudgetExceededError) as excinfo:
        budget.charge("c", cpu, 40)
    assert excinfo.value.fields["committed_bytes"] == 120
    assert excinfo.value.fields["num_saved"] == 3
    assert excinfo.value.fields["label"] == "c"


def test_exactly_at_the_budget_is_allowed() -> None:
    """The budget is a ceiling, not a strict bound: equal is fine."""

    budget = SaveBudget.from_option(100)
    assert budget is not None
    budget.charge("a", torch.device("cpu"), 100)
    assert budget.tripped is False


def test_budgets_are_tracked_per_device() -> None:
    """One device's spend must not consume another device's budget."""

    budget = SaveBudget.from_option(100)
    assert budget is not None
    # 80 bytes on each of two devices: 160 total, but neither device crosses its
    # own 100-byte ceiling, so nothing trips. A single shared ledger would.
    budget.charge("a", torch.device("cpu"), 80)
    budget.charge("b", torch.device("meta"), 80)
    assert budget.ledgers["cpu"].committed_bytes == 80
    assert budget.ledgers["meta"].committed_bytes == 80
    assert budget.tripped is False


def test_zero_byte_payloads_are_not_counted() -> None:
    """Empty payloads neither trip the budget nor inflate the saved count."""

    budget = SaveBudget.from_option(1)
    assert budget is not None
    budget.charge("a", torch.device("cpu"), 0)
    assert budget.ledgers == {}


# ---------------------------------------------------------------------------
# Portability: a session-time knob, never a portable fact
# ---------------------------------------------------------------------------


def test_save_budget_is_a_session_knob_not_a_portable_field() -> None:
    """The budget describes this process, so it must not enter the schema."""

    from torchlens._io import FieldPolicy
    from torchlens.constants import MODEL_LOG_FIELD_ORDER

    assert Trace.FIELD_POLICY["save_budget"].portable_policy is FieldPolicy.DROP
    assert Trace.FIELD_POLICY["_save_budget_accountant"].portable_policy is FieldPolicy.DROP
    assert "save_budget" not in MODEL_LOG_FIELD_ORDER
    assert "_save_budget_accountant" not in MODEL_LOG_FIELD_ORDER


def test_saved_trace_round_trips_with_the_default_budget(tmp_path: Path) -> None:
    """A loaded artifact restores the default rather than a stale ceiling."""

    trace = tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=None))
    assert trace.save_budget is None

    path = tmp_path / "trace.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    # DROP means the ceiling is not round-tripped: a loaded trace retains nothing,
    # so it is restored to the default rather than to this session's setting.
    assert loaded.save_budget == "auto"
