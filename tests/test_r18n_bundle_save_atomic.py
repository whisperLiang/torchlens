"""Regression tests for atomic ``Bundle.save(overwrite=True)`` (r18n / H1).

A failed overwrite must never destroy the user's existing good bundle. The
writer moves the old bundle aside to a sibling backup, swaps the freshly
written bundle into place, and only then removes the backup; any error before
or during the final swap leaves the OLD bundle recoverable (and restored to
its path). A clean overwrite must still replace the old bundle correctly.

Fail-before (buggy ``rmtree``-then-``rename`` tail): the injected final-rename
failure destroys the old bundle -- ``target.exists()`` is False.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("safetensors")

import torch
from torch import nn

import torchlens as tl


class _TinyBundleModel(nn.Module):
    """Minimal model used to build deterministic bundle members."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""

        return self.lin(x)


def _build_bundle(seed: int) -> "tl.Bundle":
    """Build a one-member bundle whose member name encodes ``seed``.

    Parameters
    ----------
    seed:
        Determines both the member name (``mSEED``) and the RNG state so the
        two bundles under test are distinguishable after a round-trip.

    Returns
    -------
    tl.Bundle
        A single-member bundle.
    """

    torch.manual_seed(seed)
    model = _TinyBundleModel()
    x = torch.randn(2, 4)
    member = tl.trace(model, x, layers_to_save="all", random_seed=seed)
    return tl.Bundle({f"m{seed}": member})


def _sibling_temp_dirs(target: Path) -> list[str]:
    """Return leftover ``tmp.*``/``tmp.bak.*`` siblings of ``target``.

    Parameters
    ----------
    target:
        Bundle directory whose parent is scanned for orphaned temp/backup dirs.

    Returns
    -------
    list[str]
        Sorted names of stray temp or backup directories (empty when clean).
    """

    return sorted(
        entry.name
        for entry in target.parent.iterdir()
        if entry != target and entry.name.startswith("tmp.")
    )


def test_overwrite_failure_preserves_existing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error during the final swap must NOT lose the old bundle."""

    target = tmp_path / "bundle.tlspec"
    _build_bundle(0).save(target, overwrite=True)
    assert list(tl.load(target).members) == ["m0"]

    real_rename = os.rename

    def failing_rename(src: object, dst: object, *args: object, **kwargs: object) -> None:
        # Fail ONLY the forward swap ``tmp.<hex> -> target``; allow the
        # recovery restore ``tmp.bak.<hex> -> target`` so the transient error
        # does not also block rollback.
        src_name = os.path.basename(os.fspath(src))  # type: ignore[arg-type]
        if (
            os.fspath(dst) == os.fspath(target)  # type: ignore[arg-type]
            and src_name.startswith("tmp.")
            and not src_name.startswith("tmp.bak.")
        ):
            raise OSError("injected final rename failure")
        return real_rename(src, dst, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "rename", failing_rename)
    with pytest.raises(OSError, match="injected final rename failure"):
        _build_bundle(1).save(target, overwrite=True)
    monkeypatch.undo()

    # The OLD bundle survived and is still a loadable, unchanged bundle.
    assert target.exists()
    assert list(tl.load(target).members) == ["m0"]
    # No orphaned temp/backup directories leaked.
    assert _sibling_temp_dirs(target) == []


def test_overwrite_success_replaces_bundle(tmp_path: Path) -> None:
    """A clean overwrite still swaps the new bundle in and removes the old."""

    target = tmp_path / "bundle.tlspec"
    _build_bundle(0).save(target, overwrite=True)
    assert list(tl.load(target).members) == ["m0"]

    _build_bundle(1).save(target, overwrite=True)

    assert target.exists()
    assert list(tl.load(target).members) == ["m1"]
    # Backup of the previous bundle was removed after the successful swap.
    assert _sibling_temp_dirs(target) == []


def test_overwrite_false_on_existing_raises(tmp_path: Path) -> None:
    """``overwrite=False`` on an existing target refuses and preserves it."""

    target = tmp_path / "bundle.tlspec"
    _build_bundle(0).save(target, overwrite=True)

    with pytest.raises(FileExistsError):
        _build_bundle(1).save(target, overwrite=False)

    # The refusal left the original bundle exactly as it was.
    assert list(tl.load(target).members) == ["m0"]
    assert _sibling_temp_dirs(target) == []
