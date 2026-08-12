"""Test-side inverse oracle adapter scaffolding (design-of-record 6.3).

``op_event_from_record`` feeds the UNCHANGED capture-oracle characterizer: a
compat ``OpEvent`` in, the same object out (identity); a decomposed
``OpRecord`` in (P1+), a reconstructed genuine ``OpEvent`` out. The adapter
dies in S15 with ``OpEvent``. P0 ships the seam and the identity leg so the
oracle goldens are provably insensitive to the adapter's insertion.
"""

from __future__ import annotations

from typing import Any

from torchlens.ir.events import OpEvent


def op_event_from_record(record: Any) -> OpEvent:
    """Return a genuine compat ``OpEvent`` for one journal record."""

    if isinstance(record, OpEvent):
        return record
    raise NotImplementedError(
        "OpRecord -> OpEvent inverse adaptation lands in P1 with torchlens/ir/op_record.py"
    )
