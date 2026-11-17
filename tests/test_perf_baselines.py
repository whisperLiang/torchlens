"""Nightly prepares both immutable baselines before measuring performance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_perf_baselines import prepare_baselines


def _git(repo: Path, *args: str) -> str:
    """Run git in a disposable test repository."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a two-commit repository with an initial fixed anchor."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Perf Test")
    _git(repo, "config", "user.email", "perf@example.invalid")
    (repo / "model.txt").write_text("initial\n")
    _git(repo, "add", "model.txt")
    _git(repo, "commit", "-qm", "initial anchor")
    anchor = _git(repo, "rev-parse", "HEAD")
    (repo / "model.txt").write_text("current\n")
    _git(repo, "commit", "-qam", "current")
    (repo / "anchor.json").write_text(
        json.dumps({"commit": anchor, "reason": "Reviewed initial baseline"})
    )
    return repo, anchor, _git(repo, "rev-parse", "HEAD")


def test_no_release_uses_recorded_sha_and_keeps_both_arms(history, tmp_path: Path) -> None:
    """Tagless forks measure parent/current and fixed-anchor/current."""
    repo, anchor, current = history
    out = tmp_path / "baselines"
    evidence = prepare_baselines(repo, out, repo / "anchor.json")
    assert evidence["anchor_commit"] == anchor
    assert evidence["parent_commit"] == anchor
    assert evidence["current_commit"] == current
    assert evidence["anchor_source"] == "initial_commit"
    assert "Reviewed initial baseline" in evidence["anchor_reason"]
    assert _git(out / "perf-base", "rev-parse", "HEAD") == anchor
    assert _git(out / "perf-anchor", "rev-parse", "HEAD") == anchor
    assert (out / "perf-anchor" / "model.txt").read_text() == "initial\n"
    assert json.loads((out / "perf-baselines.json").read_text()) == evidence


def test_reachable_release_tag_takes_precedence(history, tmp_path: Path) -> None:
    """A release anchor moves only when a reachable release is tagged."""
    repo, anchor, _ = history
    _git(repo, "tag", "v3.0.3", anchor)
    (repo / "anchor.json").write_text("unneeded when a release exists")
    evidence = prepare_baselines(repo, tmp_path / "baselines", repo / "anchor.json")
    assert evidence["anchor_source"] == "release_tag"
    assert evidence["anchor_ref"] == "v3.0.3"
    assert evidence["anchor_commit"] == anchor


@pytest.mark.parametrize("invalid", ["HEAD^", "0" * 40])
def test_invalid_anchor_refuses_before_creating_worktrees(
    history, tmp_path: Path, invalid: str
) -> None:
    """An invalid or unavailable anchor never degrades to parent-only measurement."""
    repo, _, _ = history
    (repo / "anchor.json").write_text(json.dumps({"commit": invalid, "reason": "test"}))
    out = tmp_path / "baselines"
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        prepare_baselines(repo, out, repo / "anchor.json")
    assert not (out / "perf-base").exists()
    assert not (out / "perf-anchor").exists()


def test_missing_blob_refuses_before_creating_worktrees(history, tmp_path: Path) -> None:
    """A commit object alone is insufficient: its checkout content must exist."""
    repo, anchor, _ = history
    blob = _git(repo, "rev-parse", f"{anchor}:model.txt")
    (repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    out = tmp_path / "baselines"
    with pytest.raises(subprocess.CalledProcessError):
        prepare_baselines(repo, out, repo / "anchor.json")
    assert not (out / "perf-base").exists()
    assert not (out / "perf-anchor").exists()
