"""Governance for the mutation driver's known-red DESELECT ledger.

b9-opus R71r3-F1 (part 2): for the second consecutive hunt pass, the driver's
``DESELECT`` list carried entries whose tests had gone GREEN on tip — each
stale entry silently deletes the margin of whatever mutants its tests would
have killed, and the driver's own "KEEP THIS LIST SHORT AND DATED" docstring
was being violated by entries added the previous wave. Manual sweeps do not
converge; this is the EXPIRY MECHANISM: every deselected node is executed
here, and a node that PASSES fails this test until its entry is removed.

Vacuously green when the list is empty (the intended steady state).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver_deselects() -> list[str]:
    """Return the driver's DESELECT node ids without importing test machinery."""

    namespace: dict[str, object] = {}
    source = (_REPO_ROOT / "tests" / "support" / "mutation_driver.py").read_text(encoding="utf-8")
    # Execute only the DESELECT assignment: parse and pull the literal, so this
    # governance test never imports the driver's subprocess machinery.
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "DESELECT":
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "DESELECT" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
    del namespace
    raise AssertionError("mutation_driver.py lost its DESELECT ledger")


@pytest.mark.heavy
def test_deselect_ledger_entries_are_still_red() -> None:
    """A DESELECTed node that passes is a stale ledger row — remove it.

    Heavy tier: each entry costs one targeted pytest subprocess. With the
    ledger empty (steady state) this is a sub-second vacuous pass.
    """

    deselects = _driver_deselects()
    if not deselects:
        return
    stale = []
    for node in deselects:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", node, "-q", "--tb=no", "-p", "no:randomly"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=1200,
        )
        if completed.returncode == 0:
            stale.append(node)
    assert not stale, (
        "mutation_driver DESELECT entries PASSED in isolation — each stale "
        "'known red' silently deletes the kill margin of whatever mutants its "
        "test would catch. Remove these rows:\n  " + "\n  ".join(stale)
    )
