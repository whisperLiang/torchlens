"""Tier-marker lint: keep the smoke tier honest.

Two enforcement layers back the documented test tiers (CLAUDE.md "Testing Tiers",
tests/AGENTS.md "Markers"):

1. **Marker disjointness** (collection-level). pytest markers are ADDITIVE: a test
   carrying ``smoke`` together with ``heavy`` or ``slow`` still runs under
   ``-m smoke`` -- the heavier marker does NOT remove it from the fast gate. That
   combination is always a partition mistake; the fix is dropping ``smoke`` from
   the test (per-parametrize-case marks that give DIFFERENT cases different tiers
   are fine and pass this check, because ``get_closest_marker`` sees the resolved
   per-item marks).

2. **Duration budget** (runtime tripwire). An UNMARKED slow test landing in smoke
   is invisible to any static lint, so the conftest ``pytest_runtest_makereport``
   hook records every smoke-marked test whose call phase exceeds
   ``SMOKE_DURATION_BUDGET_SECONDS``, and this file (ordered last by the conftest)
   fails the session naming each offender. The budget is ~3x the 5s partition
   threshold so load noise on a busy box does not false-trip it; a test that
   trips it belongs in ``heavy`` (5-20s) or ``slow`` (>20s).

Both checks read their data from the surrounding session, so they enforce on every
smoke/full-tier run and pass vacuously when this file is run alone.
"""

import pytest

pytestmark = pytest.mark.smoke


def test_no_smoke_test_carries_a_heavier_tier_marker(request: pytest.FixtureRequest) -> None:
    """No collected item may combine ``smoke`` with ``heavy`` or ``slow``."""

    conflicted = []
    for item in request.session.items:
        if item.get_closest_marker("smoke") is None:
            continue
        for heavier in ("heavy", "slow"):
            if item.get_closest_marker(heavier) is not None:
                conflicted.append(f"{item.nodeid} [smoke + {heavier}]")
    assert not conflicted, (
        "Tests carry `smoke` together with a heavier tier marker, so `-m smoke` "
        "still runs them despite the heavier mark. Drop `smoke` from each "
        "(markers are additive):\n  " + "\n  ".join(conflicted)
    )


def test_smoke_tests_stay_within_duration_budget(request: pytest.FixtureRequest) -> None:
    """Every smoke-marked test must finish within the tier's duration budget.

    The budget value lives in ``tests/conftest.py`` (``SMOKE_DURATION_BUDGET_SECONDS``)
    and rides along on each recorded offender -- a bare ``conftest`` import here would
    be ambiguous during full-suite collection (nested conftests share the module name).
    """

    offenders = getattr(request.session, "_tl_smoke_budget_offenders", [])
    lines = [f"{nodeid}: {duration:.1f}s (budget {budget:.0f}s)" for nodeid, duration, budget in offenders]
    assert not offenders, (
        "Smoke-marked tests exceeded the smoke-tier duration budget this session. "
        "Re-tier them (move to `heavy` for 5-20s, `slow` for >20s) or make them "
        "faster:\n  " + "\n  ".join(lines)
    )
