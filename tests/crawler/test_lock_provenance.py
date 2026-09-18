"""Acceptance gates retain the release lock's full anti-fabrication boundary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from . import support


@pytest.fixture
def lock_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy tracked release inputs into an isolated repository view.

    Parameters
    ----------
    tmp_path, monkeypatch:
        Isolated storage and fixture-owned replacement for Git's tracked inventory.

    Returns
    -------
    pathlib.Path
        Repository root with the same tracked environment files as the real checkout.
    """

    source = support.repository_root()
    tracked = support.tracked_paths(source)
    for path in tracked:
        if path.is_relative_to("menagerie/crawler/envs"):
            destination = tmp_path / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / path, destination)
    monkeypatch.setattr(support, "tracked_paths", lambda _root: tracked)
    return tmp_path


def test_attested_release_locks_are_allowed(lock_repository: Path) -> None:
    """Genuine target lock families are accepted, rather than blanket-banned."""

    assert support.crawler_lock_provenance_errors(lock_repository) == ()


@pytest.mark.parametrize(
    "relative_path",
    [
        "locks/round19-linux-64.lock",
        "locks/round19-linux-64.provenance.json",
        "specs/round21-release.virtual-packages.yml",
    ],
)
def test_missing_tracked_release_input_is_reported(
    lock_repository: Path, relative_path: str
) -> None:
    """A deleted tracked input must fail closed with a located gate diagnostic."""

    target = lock_repository / "menagerie/crawler/envs" / relative_path
    target.unlink()
    assert support.crawler_lock_provenance_errors(lock_repository) == (
        f"{target}:missing-tracked-file",
    )


@pytest.mark.parametrize(
    ("filename", "expected_error"),
    [
        ("round19-linux-64.lock", "lock-hash-mismatch"),
        ("round19-linux-64.resolved.sha256", "resolved-hash-file-mismatch"),
        ("round19-linux-64.probes.json", "probe-receipt-hash-mismatch"),
    ],
)
def test_tracked_release_bytes_cannot_drift(
    lock_repository: Path, filename: str, expected_error: str
) -> None:
    """Modified artifact bytes cannot retain an unchanged provenance attestation."""

    target = lock_repository / "menagerie/crawler/envs/locks" / filename
    target.write_bytes(target.read_bytes() + b"\n")
    if target.name.endswith(".resolved.sha256"):
        target.write_text("sha256:" + "0" * 64 + "\n", encoding="utf-8")
    errors = support.crawler_lock_provenance_errors(lock_repository)
    assert any(expected_error in error for error in errors), errors


def test_unattested_clean_create_is_refused(lock_repository: Path) -> None:
    """A matching artifact hash cannot replace the real clean-create attestation."""

    target = lock_repository / "menagerie/crawler/envs/locks/round19-linux-64.provenance.json"
    provenance = json.loads(target.read_bytes())
    provenance["clean_create"]["validated"] = False
    target.write_text(json.dumps(provenance), encoding="utf-8")
    assert any(
        "clean-create-unattested" in error
        for error in support.crawler_lock_provenance_errors(lock_repository)
    )


def test_untracked_provenance_cannot_attest_a_tracked_lock(
    lock_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local companion files do not stand in for missing committed evidence."""

    tracked = support.tracked_paths(lock_repository)
    monkeypatch.setattr(
        support,
        "tracked_paths",
        lambda _root: tuple(path for path in tracked if path.suffixes != [".provenance", ".json"]),
    )
    assert any(
        "missing-provenance" in error
        for error in support.crawler_lock_provenance_errors(lock_repository)
    )


def test_orphan_tracked_outputs_are_refused_but_local_solves_are_ignored(
    lock_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Git boundary rejects fabricated tracked outputs without policing local runs."""

    relative = Path("menagerie/crawler/envs/locks/fabricated.resolved.sha256")
    (lock_repository / relative).write_text("sha256:" + "0" * 64 + "\n", encoding="utf-8")
    assert support.crawler_lock_provenance_errors(lock_repository) == ()
    tracked = support.tracked_paths(lock_repository)
    monkeypatch.setattr(support, "tracked_paths", lambda _root: (*tracked, relative))
    assert any(
        "orphan-hand-authored-output" in error
        for error in support.crawler_lock_provenance_errors(lock_repository)
    )
