"""Time-argument proofs recognize both PRECALL and direct-CALL bytecode layouts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchlens.utils import rng

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("call_op", ["CALL_FUNCTION", "CALL_METHOD", "CALL", "PRECALL_CALL"])
@pytest.mark.parametrize(("value", "expected"), [(None, "now_read"), (0, "transform")])
@pytest.mark.parametrize("time_index", [0, 1])
def test_time_proof_across_call_layouts(
    monkeypatch: pytest.MonkeyPatch, call_op: str, value: int | None, expected: str, time_index: int
) -> None:
    """Decode the correct time slot for one- and two-argument native converters."""

    count = time_index + 1
    rows = [SimpleNamespace(opname="LOAD_CONST", argval="%Y", offset=0)] if time_index else []
    rows.append(SimpleNamespace(opname="LOAD_FAST", argval="timestamp", offset=2))
    if call_op == "PRECALL_CALL":
        rows.append(SimpleNamespace(opname="PRECALL", arg=count, offset=4))
    rows.append(
        SimpleNamespace(
            opname="CALL" if call_op == "PRECALL_CALL" else call_op, arg=count, offset=6
        )
    )
    frame = SimpleNamespace(f_lasti=6, f_code=None, f_locals={"timestamp": value}, f_globals={})
    monkeypatch.setattr(rng._dis_module, "get_instructions", lambda _code: iter(rows))
    assert rng._call_site_argcount(frame) == count
    assert rng._call_site_time_arg_proof(frame, count, time_index) == expected


@pytest.mark.parametrize(
    ("intervening_op", "op_arg"), [("PRECALL", 2), ("KW_NAMES", 1), ("BINARY_OP", 1)]
)
def test_time_proof_does_not_skip_unproven_instructions(
    monkeypatch: pytest.MonkeyPatch, intervening_op: str, op_arg: int
) -> None:
    """Only the exact matching PRECALL may be skipped, not arbitrary stack operations."""

    rows = [
        SimpleNamespace(opname="LOAD_CONST", argval=0, offset=0),
        SimpleNamespace(opname=intervening_op, arg=op_arg, offset=2),
        SimpleNamespace(opname="CALL", arg=1, offset=4),
    ]
    frame = SimpleNamespace(f_lasti=4, f_code=None, f_locals={}, f_globals={})
    monkeypatch.setattr(rng._dis_module, "get_instructions", lambda _code: iter(rows))
    assert rng._call_site_time_arg_proof(frame, 1, 0) == "unknown"
