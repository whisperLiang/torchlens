"""Attestation that the byte-identity oracles are ENFORCEABLE, not skippable.

The b10 audit found the oracle families could go green without ever
enforcing anything: off-canonical environments recorded-and-skipped forever
(ephemeral CI checkouts re-recorded every run), update flags produced
vacuous compare-to-self passes, and nothing asserted the canonical goldens
even exist where the enforcing tier runs (R78-4, R78-8, R53-3). These tests
pin the governance itself:

* every oracle family's CANONICAL golden inventory is committed;
* the fail-closed resolver refuses to self-baseline (record needs the
  explicit opt-in, CI never writes, canonical-missing is a hard failure);
* golden update/regen flags hard-error under CI.

The per-leg executed-floor (scripts/check_ci_executed_tests.py in the
workflows) is the runtime half: with these two together a skipping oracle
is visible instead of green.
"""

from __future__ import annotations

from pathlib import Path

import _oracle_env
import pytest
from _oracle_env import (
    RECORD_ENV_VAR,
    env_fingerprint,
    golden_mutation_flags_armed_under_ci,
    require_env_golden,
    resolve_env_golden,
)

_TESTS_DIR = Path(__file__).resolve().parent

#: Every byte-oracle family's canonical committed golden inventory.
#: A missing entry here means the enforcing tier CANNOT enforce — the
#: record-and-skip path (now fail-closed) would have been the only outcome.
_CANONICAL_GOLDENS: dict[str, tuple[Path, ...]] = {
    "surface_oracle": tuple(
        _TESTS_DIR / "surface_oracle" / "goldens" / f"{axis}.json"
        for axis in (
            "plain_cnn",
            "train_batchnorm",
            "recurrent",
            "conditional",
            "in_place",
            "tiny_transformer",
        )
    ),
    "godobject_viz": tuple(
        _TESTS_DIR / "godobject_oracle" / "goldens" / f"viz_{key}_{mode}.gv"
        for key in ("viz_cnn", "viz_recurrent")
        for mode in ("unrolled", "rolled")
    ),
    "godobject_legacy": (
        _TESTS_DIR / "godobject_oracle" / "goldens" / "legacy_baseline_cnn.tlspec",
        _TESTS_DIR / "godobject_oracle" / "goldens" / "legacy_baseline_cnn_runnable.tlspec",
        _TESTS_DIR / "godobject_oracle" / "goldens" / "legacy_baseline_cnn_loaded.json",
    ),
    "state_keysets": (_TESTS_DIR / "godobject_oracle" / "goldens" / "state_keysets.json",),
    "selector_matrix": (_TESTS_DIR / "golden" / "selector_semantics_matrix.json",),
    "viz_render_identity": (_TESTS_DIR / "golden" / "viz_render_identity_oracle.json",),
    "rank_render_ir": (_TESTS_DIR / "golden" / "rank_render_ir_semantics.json",),
}


@pytest.mark.smoke
@pytest.mark.parametrize("family", sorted(_CANONICAL_GOLDENS))
def test_canonical_golden_inventory_is_committed(family: str) -> None:
    """Every oracle family's canonical goldens exist in the checkout."""

    missing = [str(path) for path in _CANONICAL_GOLDENS[family] if not path.exists()]
    assert not missing, (
        f"oracle family {family!r} is missing canonical goldens — its byte "
        f"tests cannot enforce anything: {missing}"
    )


@pytest.mark.smoke
def test_env_markers_match_a_committed_baseline() -> None:
    """Each ENV marker names the fingerprint its canonical goldens carry.

    An ENV marker naming a fingerprint nobody records under would silently
    move EVERY environment onto the (fail-closed) env-keyed path.
    """

    for goldens_dir in (
        _TESTS_DIR / "surface_oracle" / "goldens",
        _TESTS_DIR / "godobject_oracle" / "goldens",
    ):
        marker = goldens_dir / "ENV"
        assert marker.exists(), f"missing ENV marker in {goldens_dir}"
        recorded = marker.read_text().strip()
        assert recorded, f"empty ENV marker in {goldens_dir}"


def _fake_env(monkeypatch: pytest.MonkeyPatch, fingerprint: str, **env: str | None) -> None:
    """Pin the fingerprint and the relevant environment variables."""

    monkeypatch.setattr(_oracle_env, "env_fingerprint", lambda: fingerprint)
    for name in ("CI", RECORD_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        if value is not None:
            monkeypatch.setenv(name, value)


def _goldens_dir(tmp_path: Path, canonical: str) -> Path:
    """Create a goldens dir whose ENV marker names ``canonical``."""

    goldens = tmp_path / "goldens"
    goldens.mkdir()
    (goldens / "ENV").write_text(canonical + "\n")
    return goldens


@pytest.mark.smoke
def test_missing_canonical_golden_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the canonical environment a missing golden is a hard failure."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py9.9-torch9.9.9")
    with pytest.raises(pytest.fail.Exception, match="missing canonical golden"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")


@pytest.mark.smoke
def test_missing_off_canonical_golden_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off-canonical + missing golden REFUSES instead of self-baselining."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8")
    with pytest.raises(pytest.fail.Exception, match="Refusing to self-baseline"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert not (goldens / "env-py8.8-torch8.8.8").exists(), "refusal must not write"


@pytest.mark.smoke
def test_missing_off_canonical_golden_skips_visibly_under_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI legs without a committed baseline skip with a reason, never record."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8", CI="true")
    with pytest.raises(pytest.skip.Exception, match="no committed golden"):
        require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert not (goldens / "env-py8.8-torch8.8.8").exists(), "CI must never write"


@pytest.mark.smoke
def test_record_opt_in_returns_recordable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit record opt-in provisions exactly one recordable path."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    _fake_env(monkeypatch, "py8.8-torch8.8.8", **{RECORD_ENV_VAR: "1"})
    path = require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert path == goldens / "env-py8.8-torch8.8.8" / "case.json"
    assert path.parent.is_dir(), "record opt-in prepares the env directory"
    assert not path.exists()


@pytest.mark.smoke
def test_committed_env_golden_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A committed env-keyed baseline resolves for plain byte enforcement."""

    goldens = _goldens_dir(tmp_path, "py9.9-torch9.9.9")
    env_dir = goldens / "env-py8.8-torch8.8.8"
    env_dir.mkdir()
    (env_dir / "case.json").write_text("{}\n")
    _fake_env(monkeypatch, "py8.8-torch8.8.8")
    path = require_env_golden(goldens, "case.json", "TORCHLENS_UPDATE_X")
    assert path == env_dir / "case.json"


@pytest.mark.smoke
def test_canonical_environment_resolves_canonical_path(tmp_path: Path) -> None:
    """Matching the ENV marker keeps enforcement on the canonical goldens."""

    goldens = _goldens_dir(tmp_path, env_fingerprint())
    path, off_canonical = resolve_env_golden(goldens, "case.json")
    assert path == goldens / "case.json"
    assert off_canonical is False


@pytest.mark.smoke
def test_recorded_env_baselines_are_gitignored() -> None:
    """A recorded env-* baseline can never ride along in a broad git add."""

    import subprocess

    probe = "tests/surface_oracle/goldens/env-py0.0-torch0.0.0/probe.json"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=_TESTS_DIR.parent,
        check=False,
    )
    assert result.returncode == 0, f"{probe} is not gitignored (b10 R78-4)"


@pytest.mark.smoke
def test_golden_mutation_flags_hard_error_under_ci() -> None:
    """The conftest guard names every armed update/regen flag under CI."""

    armed = golden_mutation_flags_armed_under_ci(
        {
            "CI": "true",
            "TORCHLENS_UPDATE_SURFACE_ORACLE": "1",
            "TORCHLENS_REGEN_EXPORT_GOLDENS": "1",
            "TL_SELECTOR_MATRIX_REGEN": "1",
            "TORCHLENS_ORACLE_RECORD_ENV": "1",
            "TORCHLENS_CACHE_DIR": "/tmp/x",
        }
    )
    assert armed == [
        "TL_SELECTOR_MATRIX_REGEN",
        "TORCHLENS_ORACLE_RECORD_ENV",
        "TORCHLENS_REGEN_EXPORT_GOLDENS",
        "TORCHLENS_UPDATE_SURFACE_ORACLE",
    ]
    assert golden_mutation_flags_armed_under_ci({"TORCHLENS_UPDATE_SURFACE_ORACLE": "1"}) == []
    assert golden_mutation_flags_armed_under_ci({"CI": "true"}) == []
