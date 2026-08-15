"""Ruff ratchet governance: the family ratchet and deferral ledger get TEETH.

R70 (round 3+4): the ``[tool.ruff.lint]`` block SELLS a monotone family
ratchet and an honest per-code deferral ledger, but both were prose — no test
pinned ``select``'s contents (a commit narrowing it passed every gate) and
every ledgered site count had drifted (SIM105 grew 32% while "deferred", B023
— the config's own "REAL bug class" — exceeded its stated ceiling). Two
mechanical locks:

1. ``select`` must remain a SUPERSET of the frozen family floor — the ratchet
   can only grow.
2. Every deferred code carries a NO-GROWTH ceiling, measured here with the
   pinned ruff over the exact CI scope in ONE combined isolated run. Debt may
   shrink (lower the ceiling in the same change — welcomed); it cannot grow.

D417 (parameter-documentation debt, R69) rides the same mechanism: it is not
ledger-deferred (the D family is not in ``select``) but its count was rotting
measurably (140 → 141 → 143 across three hunt passes) with nobody measuring.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The family ratchet floor. GROW-ONLY: entries are never removed; new
#: families join here in the same change that lands them in pyproject.
_SELECT_FAMILY_FLOOR = frozenset({"E4", "E7", "E9", "F", "I", "B", "SIM", "UP", "C4"})

#: No-growth ceilings per deferred code, measured 2026-08-15 with the pinned
#: ruff 0.15.4 over the CI scope (isolated mode + the config's extend-exclude
#: set, so per-file-ignores do not mask sites). SHRINK-ONLY: lower a ceiling
#: alongside a real cleanup; raising one is the exact silent-growth this test
#: exists to prevent (SIM105 grew 74 -> 99 while ledgered as "deferred").
_DEFERRED_CODE_CEILINGS: dict[str, int] = {
    "B905": 109,
    "B028": 2,
    "B023": 17,
    "B904": 17,
    "B018": 14,
    "B007": 16,
    "B008": 3,
    "SIM108": 88,
    "SIM105": 99,
    "SIM102": 40,
    "SIM117": 41,
    "SIM115": 7,
    "UP031": 16,
    # Not ledger-deferred (family not in select) but measurably rotting: R69's
    # parameter-documentation debt, torchlens/ only by design (tests/ has no
    # param-doc policy).
    "D417": 143,
    # The complexity family (r4 b5-fable R44r2-1 HIGH + b5-opus R44-1/2
    # convergent): completely ungated through fixwave-3 — C901 grew 408->441
    # and PLR0912 227->253 with zero tripwire while the round-1 named
    # hotspots were properly fixed. Ceilings frozen at the 2026-08-15 tip
    # measurement (pinned ruff 0.15.4, isolated defaults, torchlens/ only —
    # interior quality debt, not a test-style policy). SHRINK-ONLY like every
    # row above; splitting a hot-path god-function should lower the ceiling
    # in the same change.
    "C901": 441,
    "PLR0911": 189,
    "PLR0912": 253,
    "PLR0913": 417,
    "PLR0915": 146,
}

#: Codes measured over torchlens/ only (see the D417 and complexity notes
#: above).
_PACKAGE_ONLY_CODES = frozenset({"D417", "C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915"})

_CI_SCOPE = ("torchlens", "tests", "scripts", "tools", "benchmarks", "examples", "notebooks")


def _pyproject_text() -> str:
    """Return the pyproject.toml text."""

    return (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _configured_extend_excludes() -> list[str]:
    """Parse ``[tool.ruff] extend-exclude`` so the isolated run stays in lockstep."""

    match = re.search(
        r"^extend-exclude\s*=\s*\[(.*?)\]", _pyproject_text(), re.MULTILINE | re.DOTALL
    )
    assert match, "pyproject.toml lost [tool.ruff] extend-exclude"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_ruff_select_ratchet_never_narrows() -> None:
    """[tool.ruff.lint] select must stay a superset of the family floor."""

    match = re.search(r"^select\s*=\s*\[(.*?)\]", _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml lost [tool.ruff.lint] select entirely"
    selected = set(re.findall(r'"([^"]+)"', match.group(1)))
    missing = sorted(_SELECT_FAMILY_FLOOR - selected)
    assert not missing, (
        f"[tool.ruff.lint] select dropped ratcheted families: {missing}. The "
        "family ratchet is grow-only (R70); restore them — narrowing select is "
        "never a fix."
    )


def _count_by_code(concise_output: str) -> Counter[str]:
    """Count violations per code from ruff concise output (pure, testable)."""

    return Counter(re.findall(r"^\S+:\d+:\d+: ([A-Z]+\d+)", concise_output, re.MULTILINE))


def _measure_deferred_codes() -> Counter[str]:
    """Run the pinned ruff ONCE per scope over every ceilinged code."""

    excludes: list[str] = []
    for pattern in _configured_extend_excludes():
        excludes.extend(("--extend-exclude", pattern))
    counts: Counter[str] = Counter()
    scoped = {
        "ci-scope": [code for code in _DEFERRED_CODE_CEILINGS if code not in _PACKAGE_ONLY_CODES],
        "package-only": sorted(_PACKAGE_ONLY_CODES),
    }
    for scope_name, codes in scoped.items():
        if not codes:
            continue
        paths = _CI_SCOPE if scope_name == "ci-scope" else ("torchlens",)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                *paths,
                "--isolated",
                "--select",
                ",".join(codes),
                "--output-format",
                "concise",
                "--no-cache",
                *excludes,
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=300,
        )
        assert completed.returncode in (0, 1), (
            f"ruff invocation failed ({scope_name}): {completed.stderr[-1000:]}"
        )
        counts.update(_count_by_code(completed.stdout))
    return counts


def test_deferred_lint_debt_never_grows() -> None:
    """Every ceilinged code's measured count stays at or below its ceiling."""

    counts = _measure_deferred_codes()
    grown = [
        f"{code}: measured {counts.get(code, 0)} > ceiling {ceiling}"
        for code, ceiling in sorted(_DEFERRED_CODE_CEILINGS.items())
        if counts.get(code, 0) > ceiling
    ]
    assert not grown, (
        "deferred lint debt GREW (the ledger promises deferral, not license):\n  "
        + "\n  ".join(grown)
        + "\nFix the new sites (or, for a deliberate exception, raise the ceiling "
        "in tests/test_ruff_ratchet_governance.py with a stated reason in the "
        "same change)."
    )


def test_deferred_ceiling_parser_is_red_capable() -> None:
    """The count parser attributes planted concise output correctly."""

    planted = (
        "torchlens/a.py:1:1: B023 Function definition does not bind loop variable\n"
        "torchlens/a.py:9:5: B023 Function definition does not bind loop variable\n"
        "tests/b.py:2:3: SIM105 Use `contextlib.suppress`\n"
    )
    counts = _count_by_code(planted)
    assert counts == Counter({"B023": 2, "SIM105": 1})
    assert counts.get("B023", 0) > 1  # a ceiling of 1 would trip on this plant
