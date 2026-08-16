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


@pytest.mark.smoke
def test_every_direct_check_entry_point_is_enrolled() -> None:
    """Check-shaped functions in the direct-target scope are all mutant targets.

    b9-sol round-5 R74-2: the direct roster covered exactly one validation
    comparator and one postprocess checker while validation/core.py carried
    five more raise-on-violation entry points and postprocess/__init__.py two
    assert seams — claimed-exhaustive tripwire coverage derived from the
    metadata-contract registry alone. Registry contracts are enrolled by
    construction; this census makes the DIRECT scope structural too: a new
    ``_check_*``/``_validate_*``/``_assert_*`` def in either file must join
    MUTANTS or EXEMPT_MUTANTS (or a reasoned exclusion here) before it ships.
    """

    import re

    driver = _load_driver_module()
    enrolled = {(path, function) for path, function in driver.MUTANTS.values()} | {
        (path, function) for path, function in driver.EXEMPT_MUTANTS.values()
    }
    # Reasoned exclusions only — every entry needs a why.
    excluded: set[tuple[str, str]] = set()
    scope = ("torchlens/validation/core.py", "torchlens/postprocess/__init__.py")
    unenrolled = []
    for rel in scope:
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for name in re.findall(r"^def (_(?:check|validate|assert|verify)_\w+)", source, re.M):
            key = (rel, name)
            if key not in enrolled and key not in excluded:
                unenrolled.append(key)
    assert not unenrolled, (
        "check-shaped entry points with no mutation enrollment (their disarm "
        f"margin is unmeasured): {sorted(unenrolled)}"
    )


@pytest.mark.smoke
def test_while_exit_arms_take_the_break_operator(tmp_path: Path) -> None:
    """A raise inside a ``while`` body is disarmed with ``break``, not ``pass``.

    b9-opus R74r5-F2: the ``pass`` operator on a loop-exit arm produced a
    NON-TERMINATING mutant (the module_containment_logic cycle guard spun a
    sandbox at 99.9% CPU for 28 minutes). ``break`` disarms the raise while
    preserving termination, so the arm's margin is measurable at all.
    """

    driver = _load_driver_module()
    module = tmp_path / "checker.py"
    module.write_text(
        "class MetadataInvariantError(Exception):\n"
        "    pass\n"
        "\n"
        "def _check(chain):\n"
        "    visited = set()\n"
        "    current = 0\n"
        "    while current is not None:\n"
        "        if current in visited:\n"
        "            raise MetadataInvariantError('cycle')\n"
        "        visited.add(current)\n"
        "        current = chain.get(current)\n"
        "    if not chain:\n"
        "        raise MetadataInvariantError('empty')\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    src = module.read_text(encoding="utf-8")
    arms = driver.enumerate_raise_arms(src, "_check")
    assert len(arms) == 2
    while_keys = driver.while_exit_arm_keys(src, "_check")
    assert arms[0] in while_keys and arms[1] not in while_keys

    original = driver.neuter_raise_arm(module, "_check", 0)
    mutated_src = module.read_text(encoding="utf-8")
    assert "break  # R74-ARM-MUTANT" in mutated_src
    mutated: dict[str, object] = {}
    exec(mutated_src, mutated)  # noqa: S102 - planted fixture
    # The disarmed cycle guard TERMINATES (break) instead of spinning forever.
    assert mutated["_check"]({0: 1, 1: 0}) == "ok"
    module.write_text(original, encoding="utf-8")

    # A non-loop arm keeps the surgical ``pass`` operator.
    driver.neuter_raise_arm(module, "_check", 1)
    assert "pass  # R74-ARM-MUTANT" in module.read_text(encoding="utf-8")


def test_real_cycle_guard_arm_is_break_disarmed() -> None:
    """The actual module_containment_logic cycle arm gets the break operator."""

    import os

    from torchlens.validation.invariants import METADATA_INVARIANT_CONTRACTS

    driver = _load_driver_module()
    contract = next(c for c in METADATA_INVARIANT_CONTRACTS if c.name == "module_containment_logic")
    rel = os.path.relpath(contract.check.__code__.co_filename, _REPO_ROOT)
    src = (_REPO_ROOT / rel).read_text(encoding="utf-8")
    while_keys = driver.while_exit_arm_keys(src, contract.check.__name__)
    assert while_keys, (
        "module_containment_logic lost its while-body cycle arm -- if the walk "
        "was restructured, re-verify the non-terminating-mutant class (R74r5-F2)"
    )


@pytest.mark.smoke
def test_direct_targets_exist_and_are_neuterable() -> None:
    """Every MUTANTS/EXEMPT_MUTANTS row names a real function in a real file."""

    import ast as _ast

    driver = _load_driver_module()
    rows = list(driver.MUTANTS.values()) + list(driver.EXEMPT_MUTANTS.values())
    missing = []
    for rel, function in rows:
        path = _REPO_ROOT / rel
        if not path.exists():
            missing.append((rel, function, "file missing"))
            continue
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        if not any(
            isinstance(node, _ast.FunctionDef) and node.name == function for node in _ast.walk(tree)
        ):
            missing.append((rel, function, "function missing"))
    assert not missing, f"stale mutation-roster rows: {missing}"


@pytest.mark.smoke
def test_run_suite_deadline_yields_timeout_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suite run past its deadline returns the TIMEOUT sentinel, never hangs.

    b9-opus R74r5-F2: ``run_suite`` called ``subprocess.run`` with no
    ``timeout=``, so a non-terminating mutant consumed the sandbox and the
    campaign never produced a verdict for the rest of its batch.
    """

    driver = _load_driver_module()

    def _hang(*args: object, **kwargs: object):
        assert kwargs.get("timeout") == 5.0
        raise driver.subprocess.TimeoutExpired(cmd="pytest", timeout=5.0)

    monkeypatch.setattr(driver.subprocess, "run", _hang)
    result = driver.run_suite(tmp_path, "python", "gov", timeout=5.0)
    assert isinstance(result, driver.SuiteTimeout)
    assert result.seconds == 5.0


@pytest.mark.smoke
def test_core_check_roster_refuses_enrollment_drift(tmp_path: Path) -> None:
    """A core.py checker in neither ledger refuses the campaign loudly.

    b9-sol R74r5 finding 2: exhaustive-coverage claims rested on the
    metadata-contract registry alone while ``validation/core.py`` carried
    five verdict-steering checkers with no mutant. The derivation makes that
    drift a refusal, not a silent gap.
    """

    driver = _load_driver_module()
    core_dir = tmp_path / "torchlens" / "validation"
    core_dir.mkdir(parents=True)
    enrolled = "\n".join(
        f"def {func}():\n    pass\n" for _, func in driver.CORE_CHECK_MUTANTS.values()
    )
    excluded = "\n".join(f"def {func}():\n    pass\n" for func in driver.CORE_CHECK_EXCLUSIONS)
    (core_dir / "core.py").write_text(
        enrolled + excluded + "\ndef _check_brand_new_thing():\n    pass\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="_check_brand_new_thing"):
        driver.derive_core_check_roster(tmp_path)

    # Without the stray def the same tree is accepted.
    (core_dir / "core.py").write_text(enrolled + excluded, encoding="utf-8")
    driver.derive_core_check_roster(tmp_path)

    # A ledger row pointing at a vanished def is refused too.
    (core_dir / "core.py").write_text(enrolled, encoding="utf-8")
    with pytest.raises(SystemExit, match="without a core.py def"):
        driver.derive_core_check_roster(tmp_path)


def test_core_check_roster_matches_the_real_tree() -> None:
    """The live core.py passes the enrollment scan (no unledgered checkers)."""

    driver = _load_driver_module()
    driver.derive_core_check_roster(_REPO_ROOT)


@pytest.mark.smoke
def test_empty_mutant_selection_refuses_vacuous_green() -> None:
    """r7 R79 (fable b10 MED): zero selected mutants is a refusal, not a pass.

    An empty ``ids`` list used to skip the campaign loop and print ``all
    mutants KILLED`` with exit 0, so a shard-slicing bug or family-key rename
    turned the scheduled leg permanently green while scoring NOTHING.
    """

    driver = _load_driver_module()
    with pytest.raises(SystemExit, match="EMPTY MUTANT SELECTION"):
        driver.require_nonempty_selection([], family="arms", arm_shard="9/9")
    # Non-empty selections pass through untouched.
    driver.require_nonempty_selection(["m1"], family=None, arm_shard=None)
    # main() calls the floor after family filtering and shard slicing.
    source = (_REPO_ROOT / "tests" / "support" / "mutation_driver.py").read_text(encoding="utf-8")
    main_body = source.split("def main()", 1)[1]
    slice_pos = main_body.find("--arm-shard")
    floor_pos = main_body.find("require_nonempty_selection")
    assert slice_pos != -1 and floor_pos != -1 and floor_pos > slice_pos, (
        "the empty-selection floor must run AFTER family filtering and shard "
        "slicing in main(), or an empty slice still scores vacuously green"
    )


@pytest.mark.smoke
def test_mutation_workflow_rotation_contract() -> None:
    """r7 cluster 19: the rotation contract f9964208 claimed but never pinned.

    The scheduled leg's slot selection and family routing live in shell
    inside ``mutation.yml``; nothing else checks that the families it names
    exist in the driver, that the rotation can only produce shards 1..4 of
    4, or that step outputs stay routed through ``env:`` (the zizmor
    template-injection class). Pin all three so a workflow edit that breaks
    the campaign's selection contract goes red here instead of scoring an
    empty (now refused) or wrong slice on a scheduled Sunday.
    """

    import re

    workflow = (_REPO_ROOT / ".github" / "workflows" / "mutation.yml").read_text(encoding="utf-8")
    driver = _load_driver_module()

    # Every family literal the workflow can route exists in the driver's
    # vocabulary ("bounded" is the workflow-side fan-out alias).
    families_in_driver = set(driver.FAMILIES) if hasattr(driver, "FAMILIES") else None
    bounded_loop = re.search(r"for fam in ([a-z ]+);", workflow)
    assert bounded_loop is not None, "mutation.yml lost its bounded fan-out loop"
    workflow_families = set(bounded_loop.group(1).split())
    assert workflow_families == {"registry", "checks", "corechecks", "blocks", "exempt"}
    if families_in_driver is not None:
        assert workflow_families <= families_in_driver

    # The rotation arithmetic yields shard I/4 with I in 1..4 for every ISO
    # week (%V is 01..53); shard 0 or an out-of-range index is impossible.
    assert re.search(r"%\s*4\s*\+\s*1", workflow), (
        "rotation slot arithmetic changed: the ISO-week mapping must stay "
        "modulo-4 plus one (shards 1..4, never 0)"
    )
    assert 'shard="${slot}/4"' in workflow
    for week in range(1, 54):
        slot = week % 4 + 1
        assert 1 <= slot <= 4

    # Step outputs reach the run block through env, never inline ${{ }}
    # interpolation (r7 R82 template-injection fix).
    run_step = workflow.split("Run mutation campaign", 1)[1]
    assert "SLOT_FAMILY: ${{ steps.slot.outputs.family }}" in run_step
    assert "${{ steps.slot.outputs" not in run_step.split("run: |", 1)[1], (
        "mutation.yml interpolates step outputs directly into the shell "
        "again -- route them through env (template-injection class)"
    )

    # Dispatch inputs are validated before touching $GITHUB_OUTPUT.
    assert re.search(r"case \"\$INPUT_FAMILY\" in", workflow), (
        "dispatch input validation removed from the slot step"
    )
