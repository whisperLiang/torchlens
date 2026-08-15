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


def _load_driver_module():
    """Import the driver as a module (its import has no side effects)."""

    import importlib.util

    path = _REPO_ROOT / "tests" / "support" / "mutation_driver.py"
    spec = importlib.util.spec_from_file_location("_mutation_driver_gov", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.smoke
def test_arm_derivation_addresses_every_invariant_raise() -> None:
    """One arm mutant per ``raise MetadataInvariantError`` in every contract.

    b9-opus R74r4-F1/F2: the sub-check family was a hand-list of ONE while
    161 raise arms existed, and two single-arm disarms were PROVEN silent
    survivors. The per-arm roster is DERIVED (like the registry family), so
    enrollment drift is impossible by construction — this pins that the
    derivation reaches every contract's arms on the real tree, including the
    two proven-survivor arms by their stable ids.
    """

    import os

    from torchlens.validation.invariants import METADATA_INVARIANT_CONTRACTS

    driver = _load_driver_module()
    registry = {}
    for contract in METADATA_INVARIANT_CONTRACTS:
        code = contract.check.__code__
        registry[contract.name] = (
            os.path.relpath(code.co_filename, _REPO_ROOT),
            contract.check.__name__,
        )
    arms = driver.derive_arm_mutants(_REPO_ROOT, registry)
    # Every raise arm in every registered checker is one enrolled mutant.
    for contract, (rel, func) in registry.items():
        src = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        expected = len(driver.enumerate_raise_arms(src, func))
        enrolled = sum(1 for mid in arms if mid.split("#", 1)[0] == contract)
        assert enrolled == expected, (
            f"{contract}: {expected} raise arms in {func} but {enrolled} enrolled"
        )
    # The two b9-opus R74r4-F1 PROVEN survivors are addressable mutants; their
    # dedicated killers live in tests/test_validation.py
    # (test_corruption_functionless_sentinel_on_computational_op,
    # test_corruption_capture_witnessed_slot_permutation_is_rejected).
    survivors = [
        mid for mid in arms if mid.split("#", 1)[0] in ("op_log_fields", "capture_edge_survival")
    ]
    assert survivors, "the proven-survivor contracts lost their arm enrollment"
    sentinel_arm = arms.get("op_log_fields#a06")
    assert sentinel_arm is not None, "op_log_fields lost the functionless-sentinel arm id"
    src = (_REPO_ROOT / sentinel_arm[0]).read_text(encoding="utf-8")
    lineno, end, _ = driver.enumerate_raise_arms(src, sentinel_arm[1])[sentinel_arm[2]]
    arm_text = " ".join(line.strip() for line in src.splitlines()[lineno - 1 : end])
    assert "functionless" in arm_text, (
        "op_log_fields#a06 no longer addresses the functionless-sentinel arm — "
        "arms moved; re-derive the survivor ids in this test and re-score them"
    )


@pytest.mark.smoke
def test_arm_operator_disarms_exactly_one_arm(tmp_path: Path) -> None:
    """``neuter_raise_arm`` silences the targeted arm and ONLY that arm.

    Red-capability for the operator itself: the mutant this driver family
    plants must reproduce the b9-opus R74r4-F1 probe shape (one raise
    replaced by ``pass``, every other statement still live), otherwise a
    campaign's SURVIVOR/KILLED verdicts measure the wrong thing.
    """

    driver = _load_driver_module()
    module = tmp_path / "checker.py"
    module.write_text(
        "class MetadataInvariantError(Exception):\n"
        "    pass\n"
        "\n"
        "def _check(v):\n"
        "    if v == 1:\n"
        "        raise MetadataInvariantError('arm zero')\n"
        "    if v == 2:\n"
        "        raise MetadataInvariantError(\n"
        "            'arm one',\n"
        "        )\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    arms = driver.enumerate_raise_arms(module.read_text(encoding="utf-8"), "_check")
    assert len(arms) == 2
    original = driver.neuter_raise_arm(module, "_check", 1)
    mutated: dict[str, object] = {}
    exec(module.read_text(encoding="utf-8"), mutated)  # noqa: S102 - planted fixture
    check = mutated["_check"]
    with pytest.raises(Exception, match="arm zero"):
        check(1)  # the untargeted arm still fires
    assert check(2) == "ok"  # the targeted arm is silenced
    module.write_text(original, encoding="utf-8")
    restored: dict[str, object] = {}
    exec(module.read_text(encoding="utf-8"), restored)  # noqa: S102 - planted fixture
    with pytest.raises(Exception, match="arm one"):
        restored["_check"](2)


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
