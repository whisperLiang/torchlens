"""R45: `_state` raw-access exemption is declared AND pinned no-growth.

``torchlens._state`` is the sanctioned dependency-free toggle substrate: direct
reads of its published globals from other torchlens modules (including the bare
hot-path wrapper loads) are the documented design, exempt-by-declaration from
private-member lint ratchets (disputed-r2 b5 #3 — the ~180-site read migration
is explicitly rejected). This gate is the exemption's counterweight: the
cross-module access-site count may only SHRINK. New code should use the atomic
``_state.active_capture()`` snapshot or a state-owned transition function; a
new raw reach-in that grows the count is a reviewed contract diff (lower the
baseline when sites are removed; never raise it silently).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Module-wide smoke dropped (r3settle2 budget lint): the repo-wide ratchet
# scan below measures over the 5s smoke partition; per-test marks.

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "torchlens"

# Pinned 2026-08-14 (disputed-r2 fix wave): 292 access sites across 45 modules.
# Shrink-only: update DOWNWARD when refactors remove sites; growing it requires
# naming why the new site cannot use active_capture() or a state transition.
# 292 -> 294 (2026-08-15 r3settle reconcile): tensor_utils.py's defer-prune
# owner-thread guard reads `_active_trace`/`_active_owner_thread_id` raw on
# the hot path (r43 cross-thread pause guard -- there is no capture handle to
# route through), and validation/_pristine.py's pristine-oracle window is
# inherently wrap-state surgery (`_is_decorated` + the detector/witness mode
# reads that re-arm after unwrap/rewrap).
# 294 -> 291 (2026-08-15 fixwave-4 settle): release_model's held-ref
# normalization (capcache-r5) had added six raw unwrap-ledger reads; they now
# route through the state-owned `_state.wrap_epoch_ledgers()` accessor.
_ACCESS_SITE_BASELINE = 291


def _state_access_sites() -> list[tuple[str, int]]:
    """Return every ``_state._*`` attribute access site outside ``_state.py``."""

    sites: list[tuple[str, int]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        if rel == "torchlens/_state.py":
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and isinstance(node.value, ast.Name)
                and node.value.id == "_state"
            ):
                sites.append((rel, node.lineno))
    return sites


@pytest.mark.heavy
def test_state_raw_access_count_never_grows() -> None:
    """Cross-module ``_state._*`` access sites stay at or below the baseline."""

    sites = _state_access_sites()
    assert len(sites) <= _ACCESS_SITE_BASELINE, (
        f"{len(sites)} raw _state._* access sites exceed the pinned no-growth "
        f"baseline of {_ACCESS_SITE_BASELINE}. New code should read through "
        "_state.active_capture() or route the mutation through a state-owned "
        "transition; if a raw site is genuinely required, review it and raise "
        f"the baseline explicitly. Newest sites: {sorted(sites)[-5:]}"
    )


@pytest.mark.smoke
def test_declared_policy_and_accessor_exist() -> None:
    """The exemption declaration and the sanctioned accessor stay in lockstep."""

    from torchlens import _state

    assert "Access policy" in (_state.__doc__ or "")
    assert "active_capture" in (_state.__doc__ or "")
    trace, enabled = _state.active_capture()
    assert trace is None
    assert enabled is False


@pytest.mark.smoke
def test_active_capture_snapshot_tracks_live_session() -> None:
    """The accessor reflects an active capture and resets after it."""

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens import _state

    seen: dict[str, object] = {}

    class _Probe(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen["snapshot"] = _state.active_capture()
            return x + 1

    trace = tl.trace(_Probe(), torch.randn(2, 2))
    mid_trace, mid_enabled = seen["snapshot"]  # type: ignore[misc]
    assert mid_enabled is True
    assert mid_trace is trace
    after_trace, after_enabled = _state.active_capture()
    assert after_trace is None
    assert after_enabled is False
