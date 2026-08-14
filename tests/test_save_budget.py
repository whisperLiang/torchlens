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

import gc
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
from torchlens.fastlog import _storage_resolver
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
    """An unmeasurable device warns when automatic budgeting is disabled."""

    assert available_device_bytes(torch.device("mps")) is None
    budget = SaveBudget.from_option("auto")
    assert budget is not None
    with pytest.warns(UserWarning, match="cannot measure.*mps"):
        budget.charge("op", torch.device("mps"), 1)
    assert budget.unbudgeted_devices() == ("mps",)
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


def test_exhaustive_disk_capture_is_budgeted_until_postprocess(tmp_path: Path) -> None:
    """Default ``save='all'`` keeps RAM copies until postprocess even with disk streaming."""

    with pytest.raises(SaveBudgetExceededError):
        tl.trace(
            _model(),
            _input(),
            storage=tl.to_disk(str(tmp_path / "run.tlspec")),
            capture=CaptureOptions(save_budget=64),
        )


# ---------------------------------------------------------------------------
# The refusal itself
# ---------------------------------------------------------------------------


def test_absolute_budget_refuses_with_structured_fields() -> None:
    """The refusal carries numbers, so callers never parse the message."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1024))

    fields = excinfo.value.fields
    assert fields["budget_bytes"] == 1024
    assert fields["accounted_bytes"] > 1024
    assert fields["projected_bytes"] == fields["accounted_bytes"]
    assert fields["committed_bytes"] is None
    assert fields["device"] == "cpu"
    assert fields["num_saved"] >= 1
    assert isinstance(fields["label"], str) and fields["label"]


def test_refusal_message_names_footprint_site_and_remedies() -> None:
    """The message must explain, not just fail."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1024))
    message = str(excinfo.value)

    assert "save budget" in message
    assert "projected retained footprint" in message
    assert excinfo.value.fields["accounting_phase"] == "pre_allocation_admission"
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


def test_budget_refuses_before_the_crossing_copy_is_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission must run before ``safe_copy`` can allocate the over-budget payload."""

    allocation_attempted = False

    def fail_if_copy_runs(*args: object, **kwargs: object) -> object:
        """Record an attempted allocation and fail immediately.

        Parameters
        ----------
        *args:
            Positional arguments supplied to ``safe_copy``.
        **kwargs:
            Keyword arguments supplied to ``safe_copy``.

        Returns
        -------
        object
            Never returned.

        Raises
        ------
        AssertionError
            Always, because a successful admission guard must run first.
        """

        del args, kwargs
        nonlocal allocation_attempted
        allocation_attempted = True
        raise AssertionError("over-budget activation copy was attempted")

    monkeypatch.setattr(_storage_resolver, "safe_copy", fail_if_copy_runs)
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(
            nn.ReLU(),
            torch.randn(8),
            save=tl.func("relu"),
            capture=CaptureOptions(save_budget=1),
        )
    assert allocation_attempted is False


def test_aliasing_raw_and_transformed_payloads_are_charged_once() -> None:
    """An identity transform retains one storage allocation, not two logical fields."""

    trace = tl.trace(
        nn.ReLU(),
        torch.randn(8),
        save=tl.func("relu"),
        activation_transform=lambda tensor: tensor,
        capture=CaptureOptions(save_budget=40),
    )
    relu = next(op for op in trace.layer_list if getattr(op, "func_name", None) == "relu")
    assert relu.out is relu.transformed_out
    assert int(trace._save_budget_accountant.ledgers["cpu"].committed_bytes) == 32


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
    assert excinfo.value.fields["projected_bytes"] is None
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
# Storage identity survives pointer reuse: release credits, prune, recharge
# ---------------------------------------------------------------------------


def test_released_payload_is_credited_and_its_identity_pruned() -> None:
    """Releasing the last retained payload refunds its charge and prunes the key.

    Without pruning, a later allocation recycling the same ``data_ptr`` at the
    same size would deduplicate against the dead key and commit ZERO bytes.
    """

    budget = SaveBudget.from_option(1000)
    assert budget is not None
    payload = torch.randn(100)  # 400 bytes
    budget.commit(budget.admit("a", torch.device("cpu"), 400), (payload,))
    ledger = budget.ledgers["cpu"]
    assert ledger.committed_bytes == 400
    assert len(ledger.retained_storage) == 1
    assert ledger.num_saved == 1

    del payload
    gc.collect()
    assert ledger.committed_bytes == 0
    assert ledger.retained_storage == {}
    assert ledger.num_saved == 0


def test_recycled_storage_pointer_recharges_instead_of_committing_zero() -> None:
    """A freed-then-recycled pointer is a new storage and must be charged."""

    budget = SaveBudget.from_option(10_000)
    assert budget is not None
    cpu = torch.device("cpu")
    first = torch.randn(64)  # 256 bytes; small blocks are readily recycled
    budget.commit(budget.admit("a", cpu, 256), (first,))
    assert budget.ledgers["cpu"].committed_bytes == 256

    del first
    gc.collect()
    second = torch.randn(64)
    budget.commit(budget.admit("b", cpu, 256), (second,))
    # Whether or not the allocator recycled the exact pointer, the live retained
    # footprint is one 256-byte storage, never zero.
    assert budget.ledgers["cpu"].committed_bytes == 256
    assert budget.ledgers["cpu"].num_saved == 1


def test_shared_storage_credit_waits_for_the_last_live_alias() -> None:
    """Aliases charge once and the refund waits until every alias is released."""

    budget = SaveBudget.from_option(10_000)
    assert budget is not None
    cpu = torch.device("cpu")
    base = torch.randn(64)
    view = base[:32]
    budget.commit(budget.admit("a", cpu, 256), (base,))
    budget.commit(budget.admit("b", cpu, 128), (view,))
    ledger = budget.ledgers["cpu"]
    assert ledger.committed_bytes == 256, "one physical storage, charged once"

    del base
    gc.collect()
    assert ledger.committed_bytes == 256, "the view still pins the whole storage"

    del view
    gc.collect()
    assert ledger.committed_bytes == 0
    assert ledger.retained_storage == {}


def test_identity_transform_double_reference_credits_once() -> None:
    """The same payload committed in both slots refunds exactly once at death."""

    budget = SaveBudget.from_option(10_000)
    assert budget is not None
    payload = torch.randn(8)  # 32 bytes
    budget.commit(budget.admit("a", torch.device("cpu"), 32), (payload, payload))
    ledger = budget.ledgers["cpu"]
    assert ledger.committed_bytes == 32

    del payload
    gc.collect()
    assert ledger.committed_bytes == 0
    assert ledger.retained_storage == {}


def test_dead_accountant_does_not_break_payload_release() -> None:
    """Payloads may outlive the budget; their release callbacks must be inert."""

    budget = SaveBudget.from_option(10_000)
    assert budget is not None
    payload = torch.randn(8)
    budget.commit(budget.admit("a", torch.device("cpu"), 32), (payload,))
    del budget
    gc.collect()
    del payload  # must not raise from a stale watcher
    gc.collect()


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
