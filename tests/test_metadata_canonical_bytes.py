"""Persisted metadata bytes must not depend on ``PYTHONHASHSEED`` (B3R4-R21-2).

The M6 relation views are exact ``frozenset``s of label strings, and a
``frozenset`` pickles in hash-table iteration order — salted per process for
str elements. Two processes capturing the identical program therefore emitted
byte-different ``metadata.pkl`` for identical logical content, defeating any
artifact digest comparison. ``dump_canonical_metadata`` rewrites exact
set/frozenset reductions as ``cls(sorted_members)``.

Heavy tier: each case spawns fresh interpreters with distinct hash seeds
(the salt cannot be varied in-process).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DUMP_SCRIPT = """
import pickle, sys
labels = [f"op_{i}_{i%3}:1" for i in range(24)]
state = {
    "relations": frozenset(labels),
    "mutable": set(labels[:11]),
    "nested": {"ancestors": frozenset(labels[5:20])},
}
if sys.argv[1] == "canonical":
    from torchlens._io.scrub import dump_canonical_metadata

    class _Sink:
        def __init__(self):
            self.chunks = []

        def write(self, data):
            self.chunks.append(data)

    sink = _Sink()
    dump_canonical_metadata(state, sink)
    payload = b"".join(sink.chunks)
else:
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
sys.stdout.write(payload.hex())
"""


def _dump_bytes(mode: str, hash_seed: str) -> str:
    """Run one fresh-interpreter dump under an explicit hash seed."""

    result = subprocess.run(
        [sys.executable, "-c", _DUMP_SCRIPT, mode],
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    return result.stdout.strip()


def test_canonical_metadata_bytes_are_hashseed_independent() -> None:
    """Canonical dumps are byte-identical across hash seeds; plain pickle is not.

    The plain-pickle control proves the harness can SEE the divergence (an
    all-equal result would otherwise be vacuous), and the canonical pair
    proves the fix removes it.
    """

    control_a = _dump_bytes("plain", "7")
    control_b = _dump_bytes("plain", "14")
    assert control_a != control_b, (
        "vacuity control: plain pickle of a 24-member frozenset should differ "
        "across hash seeds; if it ever stops differing this test cannot see "
        "the defect class and must be rebuilt"
    )

    canonical_a = _dump_bytes("canonical", "7")
    canonical_b = _dump_bytes("canonical", "14")
    assert canonical_a == canonical_b
    assert canonical_a != control_a
