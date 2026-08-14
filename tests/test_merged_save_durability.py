"""Save/publish durability + hardening of the ``merged-directory`` writer
(p4 3.4 R59): permission parity, load-side symlink rejection, and the
publish-time TOCTOU re-check. No live process group needed -- rank cores are
plain captures with injected ``collective_boundary_v1`` annotations.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import torchlens as tl

# Reuse the rank-core builder from the load-identity module.
from tests.test_merged_load_identity import _rank_trace
from torchlens.merged._artifact import load_merged, save_merged
from torchlens.merged._enums import MergedErrorCode
from torchlens.merged._errors import MergedArtifactError

pytestmark = pytest.mark.smoke


def _merged():
    return tl.merge_ranks([_rank_trace(0), _rank_trace(1)])


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits only")
def test_merged_artifact_permissions_are_tight(tmp_path: Path) -> None:
    # Force a permissive umask so a missing chmod would leave group bits set.
    old = os.umask(0o002)
    try:
        art = tmp_path / "merged.tlspec"
        _merged().save(art)
    finally:
        os.umask(old)
    for rel, want_group_clear in [
        (Path("."), True),
        (Path("merge"), True),
        (Path("merge") / "descriptor.json", True),
        (Path("manifest.json"), True),
    ]:
        target = art / rel
        mode = stat.S_IMODE(target.stat().st_mode)
        assert not (mode & stat.S_IWGRP), f"{rel} is group-writable: {oct(mode)}"
        assert not (mode & stat.S_IWOTH), f"{rel} is world-writable: {oct(mode)}"


def test_load_rejects_symlinked_manifest(tmp_path: Path) -> None:
    art = tmp_path / "merged.tlspec"
    _merged().save(art)
    real = art / "manifest.json"
    moved = art / "manifest.real.json"
    real.rename(moved)
    (art / "manifest.json").symlink_to(moved)
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value
    assert "symlink" in str(excinfo.value)


def test_load_rejects_symlinked_descriptor(tmp_path: Path) -> None:
    art = tmp_path / "merged.tlspec"
    _merged().save(art)
    real = art / "merge" / "descriptor.json"
    moved = art / "merge" / "descriptor.real.json"
    real.rename(moved)
    (art / "merge" / "descriptor.json").symlink_to(moved)
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value
    assert "symlink" in str(excinfo.value)


def test_publish_time_overwrite_recheck_refuses(tmp_path: Path, monkeypatch) -> None:
    # Simulate a concurrent writer creating the target during the (long)
    # staging window: patch tree_hash to materialize ``root`` mid-save. With
    # overwrite=False the publish must refuse rather than clobber it.
    art = tmp_path / "merged.tlspec"
    merged = _merged()

    import torchlens.merged._artifact as artifact_mod

    real_tree_hash = artifact_mod.tree_hash
    created = {"done": False}

    def racing_tree_hash(member_path: Path) -> str:
        if not created["done"]:
            art.mkdir(parents=True, exist_ok=True)
            (art / "sentinel").write_text("concurrent writer")
            created["done"] = True
        return real_tree_hash(member_path)

    monkeypatch.setattr(artifact_mod, "tree_hash", racing_tree_hash)
    with pytest.raises(MergedArtifactError) as excinfo:
        save_merged(merged, art, overwrite=False)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_INPUT_INVALID.value
    # The concurrent writer's artifact is untouched.
    assert (art / "sentinel").read_text() == "concurrent writer"


def test_honest_save_round_trips(tmp_path: Path) -> None:
    art = tmp_path / "merged.tlspec"
    _merged().save(art)
    loaded = load_merged(art)
    assert loaded.value_status.value == "attested_complete"
    assert loaded.rank_ids == (0, 1)
