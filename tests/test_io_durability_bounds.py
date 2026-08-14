"""io durability + bounds hardening lane (fix/iodur, 2026-08-14).

Each test FAILS against the pre-fix behavior:

* MED1 -- the save path fsyncs every written file, the temp directory tree,
  and the parent directory around the publish rename (no fsync existed
  anywhere in ``torchlens/_io/`` before), so a power/OS crash cannot publish
  a torn artifact after the old bundle was already replaced.
* MED2 -- ``Bundle``/trace overwrite saves move the existing bundle aside
  just BEFORE the atomic swap, not at save start, so a hard crash mid-save
  (SIGKILL/power loss) never leaves the target path with no bundle at all.
* MED3 -- the legacy ``kind=bundle`` ``metadata.pkl`` load path enforces the
  same byte ceiling as the trace path.
* MED4 -- manifest tensor entries are bounded in count and duplicate
  ``blob_id`` / ``relative_path`` values refuse (eager-verify CPU
  amplification from a KB-sized hostile manifest).
* MED5 -- resaving a loaded bundle with ``include_source=False`` does not
  re-emit the loaded (possibly forged) provenance verbatim.
* MED6 -- the runnable dead-model fallback refuses a non-tensor embedded
  state entry typed instead of crashing with ``AttributeError`` mid-save.
* LOW -- NUL-byte relative paths refuse typed; manifest integer fields
  reject bools/negatives/absurd dims; tensor-entry sha256 is format-checked
  at parse; bounded JSON refuses NaN/Infinity constants.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.manifest import Manifest

pytestmark = pytest.mark.smoke


def _tiny() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def _trace() -> tl.Trace:
    return tl.trace(_tiny().eval(), torch.randn(2, 4), layers_to_save="all")


def _save(tmp_path: Path, name: str = "b.tlspec") -> Path:
    spec = tmp_path / name
    tl.save(_trace(), str(spec))
    return spec


def _recording_fsync(synced: list[Path]):
    real_fsync = os.fsync

    def recorder(fd: int) -> None:
        try:
            synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        except OSError:
            pass
        real_fsync(fd)

    return recorder


# --------------------------------------------------------------------------- #
# MED1: fsync-before-publish                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_trace_save_fsyncs_files_and_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """A trace save fsyncs its sidecars, blobs, and the parent directory.

    Fail-before: ``grep fsync torchlens/_io/`` was empty -- temp+rename survived
    a process crash but not power loss; a "successful" save could hold
    zero-length files while the old bundle was already gone.
    """

    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    spec = _save(tmp_path)
    names = {path.name for path in synced}
    assert "manifest.json" in names, "manifest.json was not fsynced before publish"
    assert "metadata.pkl" in names, "metadata.pkl was not fsynced before publish"
    assert any(path.suffix == ".safetensors" for path in synced), "no blob was fsynced"
    assert spec.parent in synced, "the publish rename was not made durable (parent dir fsync)"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_bundle_save_fsyncs_members_and_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """A Bundle save fsyncs bundle.json, nested members, and the parent dir."""

    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    target = tmp_path / "bundle.tlspec"
    tl.Bundle({"m": _trace()}).save(target, overwrite=True)
    names = {path.name for path in synced}
    assert "bundle.json" in names, "bundle.json was not fsynced before publish"
    assert "manifest.json" in names, "member manifests were not fsynced before publish"
    assert target.parent in synced, "the publish rename was not made durable (parent dir fsync)"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_manifest_write_fsyncs_the_file(tmp_path: Path, monkeypatch) -> None:
    """``Manifest.write`` (fastlog finalize path) fsyncs its own file."""

    spec = _save(tmp_path)
    manifest = Manifest.read(spec / "manifest.json")
    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    destination = tmp_path / "standalone-manifest.json"
    manifest.write(destination)
    assert destination in synced
