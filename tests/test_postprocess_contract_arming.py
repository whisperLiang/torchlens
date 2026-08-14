"""Direct liveness killers for ``_check_postprocess_contract`` (b9-sol R74-1).

The postprocess contract checker was a surviving mutant: a return-None disarm
stayed green across the mutation arming suite because the driver only ran
validation files, and the planted enforcement tests in
``tests/test_postprocess_dag.py`` need a real armed capture. These tests call
the checker DIRECTLY with synthetic violating inputs, so neutering its body is
killed in every ordinary pytest run — no audit env vars, no capture, no driver.

The checker is assert-based by design (see the executor header note), so these
tests carry ``requires_assertions`` and skip under ``python -O``.
"""

from __future__ import annotations

import pytest

from torchlens._trace_core.op_store import StepAuditResult
from torchlens.postprocess import _check_postprocess_contract

pytestmark = [pytest.mark.smoke, pytest.mark.requires_assertions]


def _audit(
    *,
    written_columns: set[str] | None = None,
    released_rows: int = 0,
    read_columns: set[str] | None = None,
    clone_read_columns: set[str] | None = None,
    effective_write_columns: set[str] | None = None,
) -> StepAuditResult:
    """Build a synthetic closed-window audit result.

    Parameters
    ----------
    written_columns, released_rows, read_columns, clone_read_columns, effective_write_columns:
        Field overrides; everything defaults to the empty observation.

    Returns
    -------
    StepAuditResult
        Synthetic audit observation for one step window.
    """

    return StepAuditResult(
        written_columns=written_columns or set(),
        released_rows=released_rows,
        read_columns=read_columns or set(),
        clone_read_columns=clone_read_columns or set(),
        effective_write_columns=effective_write_columns or set(),
    )


def test_unknown_step_contract_is_rejected() -> None:
    """A step id with no registered contract raises, never silently passes."""

    with pytest.raises(AssertionError, match="Unknown postprocess step contract"):
        _check_postprocess_contract(object(), "not-a-real-step", None)


def test_undeclared_write_is_rejected() -> None:
    """An observed write outside the step's declared write set raises.

    Step 2 declares exactly two write columns; a synthetic foreign column in
    the window's observations must trip the undeclared-write assertion.
    """

    audit = _audit(written_columns={"tl_totally_undeclared_column"})
    with pytest.raises(AssertionError, match="wrote undeclared op-store columns"):
        _check_postprocess_contract(object(), "2", audit)


def test_unsanctioned_row_release_is_rejected() -> None:
    """Whole-row releases on a step without a 'deletes' sanction raise."""

    audit = _audit(released_rows=3)
    with pytest.raises(AssertionError, match="without a 'deletes' row_effects sanction"):
        _check_postprocess_contract(object(), "2", audit)


def test_undeclared_read_is_rejected_in_enforce_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the read audit in enforce mode, an undeclared read raises."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    audit = _audit(read_columns={"tl_totally_undeclared_column"})
    with pytest.raises(AssertionError, match="read undeclared op-store columns"):
        _check_postprocess_contract(object(), "2", audit)


def test_unsanctioned_clone_read_is_rejected_in_enforce_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row-clone reads on a step without a 'creates' sanction raise."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    audit = _audit(clone_read_columns={"out"})
    with pytest.raises(AssertionError, match="row-clone reads"):
        _check_postprocess_contract(object(), "2", audit)


def test_step_postcondition_runs_on_none_audit() -> None:
    """Postconditions fire even without a window (step-0 prologue path).

    A stub trace with empty ``output_layers`` must trip step 1's
    output-registration postcondition when the audit result is ``None``.
    """

    class _Stub:
        output_layers: list[str] = []

    with pytest.raises(AssertionError, match="must register output layers"):
        _check_postprocess_contract(_Stub(), "1", None)
