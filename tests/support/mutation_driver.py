"""Tripwire mutation driver: prove the validation suite KILLS neutered checks.

Institutionalized from the b9 hunt's throwaway seeds (R74/75-5), with the
pristine-control protocol BAKED IN: an un-controlled adjudication on a tree
with baseline reds hallucinated two kills during the b9 hunt, so this driver
refuses to score mutants until the UNMUTATED suite is green in the same
sandbox.

Two mutant families, one per disarming direction:

* ``MUTANTS`` neuters a tripwire check with an unconditional ``return None``
  (the first statement after the docstring) -- the disarming direction for
  raise-on-violation invariants.
* ``EXEMPT_MUTANTS`` plants ``return True`` on the perturbation-exemption
  dispatcher and each ``_check_*_exempt`` gate (b9p3 R74p3-F2): for a
  predicate whose ``True`` means "skip the sensitivity check", ``return
  None`` is falsy and makes the tripwire STRICTER -- the dangerous direction
  is exempt-everything, so it needs its own operator.

Kill attribution is per-test, not per-exit-code (b9 R74-2: a green control
still printed a "KILLED" off an unrelated flaky red): each run's FAILED node
ids are parsed and a mutant is KILLED only by ``killers = mutant_failures -
control_failures``, with the killer node ids named in the verdict. A mutant
run that reds without any parseable failed test (collection error, crash) is
an ERROR verdict, never a kill.

Usage (from the repo root)::

    python tests/support/mutation_driver.py --make-sandbox /tmp/tl-mut M04 X01
    python tests/support/mutation_driver.py --sandbox /tmp/tl-mut/repo  # all

The driver only ever writes inside the sandbox; running against the real
checkout is refused. It is a SCRIPT, deliberately not named ``test_*``: the
red-capability *tests* live in the suite itself; this measures their margin.
It is wired into no CI leg -- the margin is measured only when invoked.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: mutant id -> (relative file, function to neuter with ``return None``).
#: Every entry names a verdict-steering invariant check; extend when a new
#: contract lands.
MUTANTS: dict[str, tuple[str, str]] = {
    "M01": ("torchlens/validation/_invariants_topology.py", "_check_graph_topology"),
    "M02": ("torchlens/validation/_invariants_payloads.py", "_check_op_log_fields"),
    "M03": ("torchlens/validation/_invariants_connectivity.py", "_check_graph_connectivity"),
    "M04": ("torchlens/validation/_invariants_connectivity.py", "_check_ancestry_closure"),
    "M05": ("torchlens/validation/_invariants_equivalence.py", "_check_loop_detection_invariants"),
    "M06": ("torchlens/validation/_invariants_equivalence.py", "_check_equivalence_symmetry"),
    "M07": ("torchlens/validation/_invariants_connectivity.py", "_check_lookup_key_consistency"),
    "M08": ("torchlens/validation/_invariants_conditional_modules.py", "_check_module_hierarchy"),
    "M09": ("torchlens/validation/_invariants_topology.py", "_check_trace_self_consistency"),
    "M10": ("torchlens/validation/_invariants_conditional_base.py", "_check_recurrence_invariants"),
    "M11": ("torchlens/validation/_invariants_equivalence.py", "_check_graph_ordering"),
    "M12": ("torchlens/validation/_invariants_modules_params.py", "_check_param_xrefs"),
    # The flagship per-op replay comparator (R74/75-2): neutering the
    # comparison result at the callsite must be killed by the corruption
    # battery, not by a single diagnostics test.
    "M13": ("torchlens/validation/core.py", "_deep_numeric_replay_matches_saved"),
    # The postprocess step-contract checker (b9-sol R74-1): a return-None
    # disarm previously survived because this suite listed only validation
    # files. Killed directly by tests/test_postprocess_contract_arming.py and
    # through real armed captures by the test_postprocess_dag enforcement
    # plants below.
    "M14": ("torchlens/postprocess/__init__.py", "_check_postprocess_contract"),
}

#: mutant id -> (relative file, exemption predicate to disarm with ``return
#: True``). ``True`` means "this layer is exempt from the perturbation
#: sensitivity check", so the dangerous direction is exempt-everything --
#: killed by the deliberately-named ``*_is_not_exempt`` negative tests
#: (hand-run on the dispatcher during b9p3: 28 killers).
EXEMPT_MUTANTS: dict[str, tuple[str, str]] = {
    "X01": ("torchlens/validation/core.py", "_check_perturbation_exemptions"),
    "X02": ("torchlens/validation/exemptions.py", "_check_getitem_exempt"),
    "X03": ("torchlens/validation/exemptions.py", "_check_setitem_exempt"),
    "X04": ("torchlens/validation/exemptions.py", "_check_index_put_exempt"),
    "X05": ("torchlens/validation/exemptions.py", "_check_lstm_exempt"),
    "X06": ("torchlens/validation/exemptions.py", "_check_interpolate_exempt"),
    "X07": ("torchlens/validation/exemptions.py", "_check_scatter_exempt"),
    "X08": ("torchlens/validation/exemptions.py", "_check_one_arg_where_index_exempt"),
    "X09": ("torchlens/validation/exemptions.py", "_check_where_exempt"),
    "X10": ("torchlens/validation/exemptions.py", "_check_masked_fill_exempt"),
    "X11": ("torchlens/validation/exemptions.py", "_check_norm_running_stat_exempt"),
    "X12": ("torchlens/validation/exemptions.py", "_check_scatter_or_index_domain_exempt"),
    "X13": ("torchlens/validation/_invariants_backward_flow.py", "_is_func_call_id_exempt"),
    # Landed after the b9p3 inventory (R08 exemption-narrowing wave); swept in
    # so the newest gate is margin-measured like its siblings.
    "X14": ("torchlens/validation/exemptions.py", "_check_zipped_sibling_exempt"),
}

#: mutant id -> the planted return value (the family's disarming direction).
OPERATORS: dict[str, str] = {
    **dict.fromkeys(MUTANTS, "None"),
    **dict.fromkeys(EXEMPT_MUTANTS, "True"),
}

ALL_MUTANTS: dict[str, tuple[str, str]] = {**MUTANTS, **EXEMPT_MUTANTS}

#: Bounded arming suite: the files whose job is to kill the mutants above.
SUITE = [
    "tests/test_validation.py",
    "tests/test_replay_corruption_battery.py",
    "tests/test_internals.py",
    "tests/test_ancestry_closure_invariant.py",
    "tests/test_conditional_invariants.py",
    "tests/test_loop_synthesis_ground_truth.py",
    "tests/test_r29_capval_hardening.py",
    # M14 killers: the direct synthetic-input liveness file plus the two
    # armed-capture enforcement plants (undeclared read, in-place write
    # smuggle) that exercise the checker through a real postprocess run.
    "tests/test_postprocess_contract_arming.py",
    "tests/test_postprocess_dag.py::test_read_enforcement_trips_on_undeclared_read",
    "tests/test_postprocess_dag.py::test_executor_seam_patched_step_executes_and_audits",
    # The dedicated arming file for the newest oracle-independence witnesses
    # (edge-occurrence multiplicity et al.) -- its ABSENCE let a sub-check
    # deletion survive the whole SUITE while this file's killer caught it in
    # 1.4s (b9-opus R74r3-F2 part 1).
    "tests/test_oracle_independence.py",
]

#: Known baseline reds, deselected so a mutant verdict is never confounded.
#: KEEP THIS LIST SHORT AND DATED; every entry weakens the margin measurement
#: for whatever its tests would have killed. (The two ancestry-closure
#: deselects were removed 2026-08-15: both tests are green on tip and they
#: are M04's natural killers -- a stale entry here silently deleted M04's
#: margin. The two capture-r3 session-isolation deselects were removed later
#: the same day: green on tip for the second consecutive pass.)
#:
#: EXPIRY IS ENFORCED: tests/test_mutation_driver_governance.py runs every
#: entry and FAILS when a deselected node passes, so a stale entry can no
#: longer silently delete a mutant's margin (b9-opus R71r3-F1 part 2 -- the
#: second consecutive pass violated the keep-short-and-dated rule).
DESELECT: list[str] = []

#: Directories/patterns a sandbox never needs (b9p3 R74p3-F3: without these
#: --make-sandbox copied 6.0 GB -- 5.3 GB .venv + 127 MB menagerie -- vs
#: ~450 MB with them; --python supplies the interpreter, so the sandbox
#: needs no venv).
SANDBOX_IGNORE = (
    ".git",
    "__pycache__",
    ".ruff_cache",
    "*.egg-info",
    ".venv",
    "menagerie",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    "*.tlspec",
)


def neuter(path: Path, func: str, value: str) -> str:
    """Insert an early ``return <value>`` into ``func`` and return the original text.

    Parameters
    ----------
    path:
        File containing the function.
    func:
        Function name to neuter (first match wins).
    value:
        Source expression for the planted return value (the family's
        disarming direction: ``"None"`` for invariant checks, ``"True"``
        for exemption predicates).

    Returns
    -------
    str
        The file's original source, for restoration.
    """

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func
        ),
        None,
    )
    if target is None:
        raise SystemExit(f"function {func} not found in {path}")
    body = target.body
    first = body[0]
    anchor = (
        body[1]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and len(body) > 1
        else first
    )
    lines = src.splitlines(keepends=True)
    lines.insert(anchor.lineno - 1, f"{' ' * anchor.col_offset}return {value}  # R74-MUTANT\n")
    path.write_text("".join(lines), encoding="utf-8")
    return src


def parse_failures(stdout: str) -> frozenset[str]:
    """Extract failed/errored test node ids from a ``-rf -q`` pytest run.

    Parameters
    ----------
    stdout:
        Captured pytest stdout.

    Returns
    -------
    frozenset[str]
        Node ids reported ``FAILED`` or ``ERROR`` in the short summary.
    """

    failed = set()
    for line in stdout.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            node = line.split(" ", 2)[1]
            failed.add(node.split(" - ", 1)[0])
    return frozenset(failed)


def run_suite(sandbox: Path, python: str, tag: str) -> subprocess.CompletedProcess:
    """Run the bounded arming suite inside the sandbox.

    No ``-x``: kill attribution needs the FULL failed set of every run, both
    to name each mutant's killers and to measure the margin (killer count),
    not just first-red-wins.

    Parameters
    ----------
    sandbox:
        Sandbox repo root.
    python:
        Python executable to run pytest with.
    tag:
        Unique tag for basetemp/cache isolation.

    Returns
    -------
    subprocess.CompletedProcess
        The finished pytest process.
    """

    cache = sandbox / f".cache-{tag}"
    cache.mkdir(exist_ok=True)
    cmd = [python, "-m", "pytest", *SUITE, "-p", "no:randomly", "-q", "--tb=no", "-rf"]
    cmd += ["--basetemp", str(sandbox / f".bt-{tag}")]
    for node in DESELECT:
        cmd += ["--deselect", node]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
        TORCHLENS_CACHE_DIR=str(cache),
    )
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=sandbox)


def main() -> None:
    """Parse arguments, enforce the pristine control, and score each mutant."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutants", nargs="*", default=[], help="mutant ids (default: all)")
    parser.add_argument("--sandbox", type=Path, help="existing sandbox repo root")
    parser.add_argument(
        "--make-sandbox",
        type=Path,
        help="copy the current repo to DIR/repo and use it as the sandbox",
    )
    parser.add_argument("--python", default=sys.executable, help="python to run pytest with")
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="UNSAFE: skip the pristine control (only when just proven green)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if args.make_sandbox:
        sandbox = args.make_sandbox / "repo"
        if not sandbox.exists():
            print(f"copying {repo} -> {sandbox} ...", flush=True)
            shutil.copytree(repo, sandbox, ignore=shutil.ignore_patterns(*SANDBOX_IGNORE))
    else:
        sandbox = args.sandbox
    if sandbox is None:
        raise SystemExit("need --sandbox DIR or --make-sandbox DIR")
    sandbox = sandbox.resolve()
    if sandbox == repo:
        raise SystemExit("refusing to mutate the real checkout; use --make-sandbox")

    ids = args.mutants or sorted(ALL_MUTANTS)
    unknown = [mid for mid in ids if mid not in ALL_MUTANTS]
    if unknown:
        raise SystemExit(f"unknown mutant ids: {unknown}")

    # Pristine control: verdicts are meaningless over a red baseline (the b9
    # hunt's un-controlled pass hallucinated 2 kills off pre-existing reds).
    control_failures: frozenset[str] = frozenset()
    if not args.skip_control:
        control = run_suite(sandbox, args.python, "control")
        control_failures = parse_failures(control.stdout)
        if control.returncode != 0:
            named = "\n".join(sorted(control_failures)) or "\n".join(
                control.stdout.strip().splitlines()[-8:]
            )
            raise SystemExit(
                "PRISTINE CONTROL RED -- fix or deselect the baseline before "
                f"scoring any mutant:\n{named}"
            )
        print("control: GREEN", flush=True)

    results: dict[str, dict[str, object]] = {}
    for mid in ids:
        rel, func = ALL_MUTANTS[mid]
        path = sandbox / rel
        original = neuter(path, func, OPERATORS[mid])
        try:
            proc = run_suite(sandbox, args.python, mid)
        finally:
            path.write_text(original, encoding="utf-8")
        failures = parse_failures(proc.stdout)
        killers = sorted(failures - control_failures)
        if proc.returncode != 0 and not failures:
            # Collection error / crash: the suite never scored the mutant.
            verdict = "ERROR"
        elif killers:
            verdict = "KILLED"
        else:
            verdict = "SURVIVOR"
        results[mid] = {
            "file": rel,
            "func": func,
            "operator": f"return {OPERATORS[mid]}",
            "returncode": proc.returncode,
            "verdict": verdict,
            "killers": killers,
            "n_killers": len(killers),
            "tail": proc.stdout.strip().splitlines()[-4:],
        }
        print(json.dumps({mid: results[mid]}), flush=True)

    survivors = sorted(mid for mid, row in results.items() if row["verdict"] == "SURVIVOR")
    errors = sorted(mid for mid, row in results.items() if row["verdict"] == "ERROR")
    print("RESULTS " + json.dumps(results))
    if errors:
        print(f"ERRORS: {errors} -- suite crashed before scoring; not a kill, not a pass")
    if survivors:
        print(f"SURVIVORS: {survivors} -- each needs a new planted-corruption test")
    if errors or survivors:
        raise SystemExit(1)
    print("all mutants KILLED")


if __name__ == "__main__":
    main()
