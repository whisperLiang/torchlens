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

import json
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import bundle as bundle_mod, manifest as manifest_mod
from torchlens._io.manifest import Manifest
from torchlens.errors import TorchLensIOError

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


# --------------------------------------------------------------------------- #
# MED2: overwrite aside-rename happens at the swap, not at save start          #
# --------------------------------------------------------------------------- #


def test_overwrite_save_keeps_old_bundle_at_target_until_swap(tmp_path: Path, monkeypatch) -> None:
    """The existing bundle stays AT its path for the whole write phase.

    Fail-before: the trace-save writer renamed the old bundle aside to
    ``.bak.<uuid>`` at save START, before the minutes-long scrub/blob write.
    Python exception paths restored it, but SIGKILL/power loss mid-save left
    the target with NO bundle (old data stranded under an undocumented backup
    name), and concurrent readers saw the bundle vanish for the entire save.
    The SIGKILL window is the whole span between the early aside-rename and
    the swap; observing the target mid-save (at ``_build_manifest``, after
    blob writes) is the RED discriminator for that ordering.
    """

    spec = tmp_path / "b.tlspec"
    tl.save(_trace(), str(spec))
    real_build_manifest = bundle_mod._build_manifest
    observed: dict[str, bool] = {}

    def observing_build_manifest(*args, **kwargs):
        observed["target_exists_mid_save"] = spec.exists()
        observed["target_loadable_mid_save"] = (spec / "manifest.json").is_file()
        return real_build_manifest(*args, **kwargs)

    monkeypatch.setattr(bundle_mod, "_build_manifest", observing_build_manifest)
    tl.save(_trace(), str(spec), overwrite=True)
    assert observed["target_exists_mid_save"], "old bundle was moved aside at save start"
    assert observed["target_loadable_mid_save"]
    # The overwrite itself still completed and left no backup debris.
    tl.load(str(spec))
    assert not list(tmp_path.glob("*.bak.*"))


# --------------------------------------------------------------------------- #
# MED3: legacy kind=bundle metadata.pkl byte ceiling                           #
# --------------------------------------------------------------------------- #


def test_legacy_bundle_metadata_pkl_ceiling_refuses_oversize(tmp_path: Path, monkeypatch) -> None:
    """The legacy bundle branch enforces the same pkl byte ceiling as traces.

    Fail-before: ``_load_unified_bundle``'s ``kind=bundle`` legacy branch fed
    ``metadata.pkl`` straight into the unpickler with no fstat cap (the trace
    path had one), so an absurd on-disk pickle was an alloc/time DoS at
    ``tl.load``. Pre-fix this raised the unrelated "is not a Bundle" error
    only AFTER unpickling the whole payload.
    """

    legacy_dir = tmp_path / "legacy.tlspec"
    legacy_dir.mkdir()
    (legacy_dir / "metadata.pkl").write_bytes(pickle.dumps({"not": "a bundle"}))
    monkeypatch.setattr(bundle_mod, "_MAX_METADATA_PKL_BYTES", 16)
    with pytest.raises(TorchLensIOError, match="ceiling"):
        bundle_mod._load_unified_bundle(legacy_dir)


# --------------------------------------------------------------------------- #
# MED4: manifest entry-count ceiling + duplicate blob identity refusal         #
# --------------------------------------------------------------------------- #


def _saved_manifest_data(tmp_path: Path) -> dict:
    spec = _save(tmp_path)
    return json.loads((spec / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_refuses_duplicate_blob_ids(tmp_path: Path) -> None:
    """Two tensor entries sharing one blob_id refuse at parse.

    Fail-before: duplicates silently last-won in the load-side entry indexes
    while eager verification did per-entry sha256 + safetensors decode work,
    so a KB manifest with millions of same-blob entries bought hours of CPU.
    """

    data = _saved_manifest_data(tmp_path)
    data["tensors"] = [*data["tensors"], dict(data["tensors"][0])]
    with pytest.raises(TorchLensIOError, match="duplicate blob_id"):
        Manifest.from_dict(data)


def test_manifest_refuses_duplicate_relative_paths(tmp_path: Path) -> None:
    """Two tensor entries pointing at one blob file refuse at parse."""

    data = _saved_manifest_data(tmp_path)
    forged = dict(data["tensors"][0])
    forged["blob_id"] = "zzzz-forged"
    data["tensors"] = [*data["tensors"], forged]
    with pytest.raises(TorchLensIOError, match="duplicate relative_path"):
        Manifest.from_dict(data)


def test_manifest_refuses_entry_count_above_ceiling(tmp_path: Path, monkeypatch) -> None:
    """An entry list above the structural ceiling refuses before parsing."""

    data = _saved_manifest_data(tmp_path)
    assert data["tensors"], "fixture bundle must carry at least one tensor entry"
    monkeypatch.setattr(manifest_mod, "_MAX_MANIFEST_TENSOR_ENTRIES", len(data["tensors"]) - 1)
    with pytest.raises(TorchLensIOError, match="ceiling"):
        Manifest.from_dict(data)
