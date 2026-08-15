"""File-size ratchet: god files may shrink, never grow unnoticed.

R43 (4th pass): 42 files in ``torchlens/`` exceeded 2000 lines, the top five
all GROWING through fix waves (+996 lines on ``data_classes/op.py`` in ~200
commits), and no size ceiling existed anywhere — no ruff rule, no test, no
hook. This ratchet freezes the frontier:

- an UNLEDGERED module may not exceed ``_NEW_FILE_LINE_CAP`` (2000 — the
  repo's own god-file threshold; the style rule remains 800, enforced by
  review, not this backstop);
- a LEDGERED module may not exceed its frozen ceiling (current size at
  ratchet time, rounded up to the next 50 for mid-wave slack). Growing a
  ledgered file is a CONSCIOUS act: raise its ceiling in the same change,
  with a reason, and expect review pushback — the intended direction is DOWN
  (split along the seams catalogued in the R43 findings);
- a ledgered module that drops to/below the cap must LEAVE the ledger
  (two-way staleness, so the ledger cannot rot into permanent exemptions).

GENERATED modules (self-declared via a ``GENERATED`` marker on the first
docstring line, same convention as the ruff extend-exclude lockstep) are
exempt: their generator is the authority for their size.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _PROJECT_ROOT / "torchlens"

#: Hard line cap for any module not in the ledger below.
_NEW_FILE_LINE_CAP = 2000

#: Frozen ceilings for the god-file frontier (R43 census, 2026-08-15, rounded
#: up to the next 50). SHRINK-ONLY DOCTRINE: lower a ceiling freely; raising
#: one requires a stated reason in the same change. When a file drops to
#: <= 2000 lines, DELETE its row (the staleness check below enforces this).
_GOD_FILE_CEILINGS: dict[str, int] = {
    "torchlens/validation/core.py": 5200,
    "torchlens/data_classes/op.py": 5050,
    "torchlens/_io/runnable.py": 5000,
    "torchlens/utils/rng.py": 4750,
    "torchlens/visualization/collapse_optimizer.py": 4550,
    "torchlens/backends/jax/backend.py": 4350,
    "torchlens/_io/bundle.py": 4050,
    "torchlens/_io/runnable_load.py": 3850,
    "torchlens/user_funcs.py": 3700,
    "torchlens/backends/torch/backward.py": 3650,
    "torchlens/data_classes/trace.py": 3650,
    "torchlens/utils/_torch_compat.py": 3450,
    "torchlens/backends/torch/wrappers.py": 3350,
    "torchlens/backends/tinygrad/backend.py": 3300,
    "torchlens/backends/mlx/backend.py": 3250,
    "torchlens/postprocess/_contracts.py": 3200,
    "torchlens/backends/torch/model_prep.py": 3100,
    "torchlens/data_classes/module.py": 2950,
    "torchlens/visualization/auto_collapse.py": 2950,
    "torchlens/validation/exemptions.py": 2700,
    "torchlens/backends/paddle/backend.py": 2700,
    "torchlens/_runnable_state.py": 2650,
    "torchlens/capture/arg_positions.py": 2650,
    "torchlens/backends/jax/jaxpr.py": 2550,
    "torchlens/_capture_state_helpers.py": 2550,
    "torchlens/bundle/__init__.py": 2450,
    "torchlens/data_classes/layer.py": 2450,
    "torchlens/postprocess/loop_grouping_adapter.py": 2400,
    "torchlens/visualization/_render_leaf.py": 2350,
    "torchlens/visualization/_render_edges.py": 2350,
    "torchlens/options.py": 2350,
    "torchlens/intervention/save.py": 2300,
    "torchlens/_io/scrub.py": 2300,
    "torchlens/debug/_infer_input_shape.py": 2250,
    "torchlens/postprocess/ast_branches.py": 2250,
    "torchlens/visualization/_summary_internal/_builder.py": 2200,
    "torchlens/visualization/_render_dot.py": 2150,
    "torchlens/visualization/_render_nodes.py": 2150,
    "torchlens/_io/_safe_unpickle.py": 2100,
    "torchlens/visualization/_render_flow.py": 2100,
    "torchlens/capture/trace.py": 2050,
    "torchlens/backends/torch/completeness_witness.py": 2050,
}


def _is_generated_module(path: Path) -> bool:
    """Return whether a module self-declares as GENERATED (generator authority)."""

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped.startswith(('"""', "'''", 'r"""')) and "GENERATED" in stripped
    return False


def _line_counts() -> dict[str, int]:
    """Return line counts for every non-generated package module."""

    counts: dict[str, int] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if _is_generated_module(path):
            continue
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
        with path.open(encoding="utf-8") as handle:
            counts[relative] = sum(1 for _ in handle)
    return counts


def _ratchet_violations(counts: dict[str, int], ceilings: dict[str, int]) -> list[str]:
    """Return ratchet violations for a census (pure, red-capability-testable)."""

    violations = []
    for relative, lines in sorted(counts.items()):
        ceiling = ceilings.get(relative)
        if ceiling is None:
            if lines > _NEW_FILE_LINE_CAP:
                violations.append(
                    f"{relative}: {lines} lines exceeds the {_NEW_FILE_LINE_CAP}-line cap "
                    "for unledgered modules — split it; do not add a ledger row for new growth"
                )
        elif lines > ceiling:
            violations.append(
                f"{relative}: {lines} lines exceeds its frozen ceiling {ceiling} — split it "
                "(preferred) or consciously raise the ceiling with a stated reason"
            )
    return violations


def _stale_ledger_rows(counts: dict[str, int], ceilings: dict[str, int]) -> list[str]:
    """Return ledger rows that no longer describe a god file (two-way staleness)."""

    stale = []
    for relative, ceiling in sorted(ceilings.items()):
        lines = counts.get(relative)
        if lines is None:
            stale.append(f"{relative}: ledgered but no longer exists (ceiling {ceiling})")
        elif lines <= _NEW_FILE_LINE_CAP:
            stale.append(
                f"{relative}: {lines} lines is at/below the {_NEW_FILE_LINE_CAP} cap — "
                "delete its ledger row so the exemption cannot rot"
            )
    return stale


def test_no_package_module_exceeds_its_size_ceiling() -> None:
    """Every torchlens module respects the cap or its frozen ledger ceiling."""

    violations = _ratchet_violations(_line_counts(), _GOD_FILE_CEILINGS)
    assert not violations, (
        "file-size ratchet violations (R43 — god files may shrink, never grow "
        "unnoticed):\n  " + "\n  ".join(violations)
    )


def test_god_file_ledger_has_no_stale_rows() -> None:
    """A ledger row must leave when its file shrinks below the cap or vanishes."""

    stale = _stale_ledger_rows(_line_counts(), _GOD_FILE_CEILINGS)
    assert not stale, "stale god-file ledger rows:\n  " + "\n  ".join(stale)


def test_file_size_ratchet_is_red_capable() -> None:
    """Planted censuses trip each violation class (red-capability self-test)."""

    ceilings = {"torchlens/ledgered.py": 2500}
    grown_ledgered = _ratchet_violations({"torchlens/ledgered.py": 2501}, ceilings)
    assert len(grown_ledgered) == 1 and "frozen ceiling" in grown_ledgered[0]
    new_god = _ratchet_violations({"torchlens/new.py": 2001}, ceilings={})
    assert len(new_god) == 1 and "unledgered" in new_god[0]
    assert _ratchet_violations({"torchlens/ok.py": 2000}, ceilings={}) == []
    stale = _stale_ledger_rows({"torchlens/ledgered.py": 1999}, ceilings)
    assert len(stale) == 1 and "delete its ledger row" in stale[0]
    missing = _stale_ledger_rows({}, ceilings)
    assert len(missing) == 1 and "no longer exists" in missing[0]
