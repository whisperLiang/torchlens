"""grind-r8 b8 R56 trio: wrapper-hygiene fixes.

- Attribute-style ``torch.overrides.handle_torch_function`` dispatchers
  (the wrapped ``torch.nn.init.*`` four and the ``torch.sym_*`` family)
  used to present the torchlens WRAPPER to user ``__torch_function__``
  handlers because Site 8 patched only bare-name module globals (fable F1).
- Forwarding ``cache_clear`` let one call rebuild ``get_testing_overrides``
  WRAPPER-keyed, dropping the pristine original from the membership table
  (opus F1); the view now normalizes post-clear rebuilds through the ledger.
- The epoch-2 re-install silently skipped slots foreign-patched during the
  unwrapped gap (sol 1); it now warns, naming the slots.
- A failed identity-shim install stranded this attempt's shim objects in
  the live registry (sol 5 / fable 3); the failure unwind now purges them.
"""

from __future__ import annotations

import gc
import pickle
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.errors._base import TorchLensWarning

pytestmark = pytest.mark.smoke


def _ensure_wrapped() -> None:
    tl.trace(nn.Linear(2, 2), torch.randn(1, 2))


class _RecordingSubclass(torch.Tensor):
    seen: list = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.seen.append(func)
        return super().__torch_function__(func, types, args, kwargs or {})


def test_attribute_style_dispatchers_present_originals() -> None:
    """nn.init/sym-family handlers must see pre-wrap ORIGINALS (fable F1)."""

    _ensure_wrapped()
    wrapped_uniform = torch.nn.init.uniform_
    original = _state._decorated_to_orig.get(id(wrapped_uniform))
    assert original is not None, "expected nn.init.uniform_ to be wrapped"
    _RecordingSubclass.seen.clear()
    probe = torch.randn(3).as_subclass(_RecordingSubclass)
    torch.nn.init.uniform_(probe)
    assert any(func is original for func in _RecordingSubclass.seen), (
        "handler saw the torchlens wrapper, not the pre-wrap original"
    )
    assert not any(func is wrapped_uniform for func in _RecordingSubclass.seen)


def test_handle_torch_function_reexports_pickle_during_epoch() -> None:
    """The shim now lives AT torch.overrides, so by-reference pickle works."""

    _ensure_wrapped()
    pickle.dumps(F.handle_torch_function)
    pickle.dumps(torch.overrides.handle_torch_function)


def test_testing_overrides_stay_original_keyed_after_cache_clear() -> None:
    """opus F1: a post-clear rebuild must not be wrapper-keyed."""

    _ensure_wrapped()
    original = _state._decorated_to_orig.get(id(F.relu))
    assert original is not None
    table = torch.overrides.get_testing_overrides()
    assert original in table
    torch.overrides.get_testing_overrides.cache_clear()
    gc.collect()
    rebuilt = torch.overrides.get_testing_overrides()
    assert original in rebuilt, "pristine original fell out of the table post-clear"
    assert any(key is original for key in rebuilt), "raw keys are not original-keyed"
    assert not any(key is F.relu for key in rebuilt), "raw keys are wrapper-poisoned"


def test_epoch2_foreign_patch_skip_warns() -> None:
    """sol 1: the re-install skip must name foreign-patched slots.

    No ``monkeypatch`` for the slot on purpose: its teardown undo would fire
    AFTER the restoring re-wrap below and strand a raw ``torch.cos`` inside
    a wrapped epoch, polluting later wrap-state tests.
    """

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    _ensure_wrapped()
    unwrap_torch()
    original_cos = torch.cos

    def foreign_cos(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test shim
        return original_cos(*args, **kwargs)

    torch.cos = foreign_cos
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrap_torch()
        diagnostics = [
            w for w in caught if w.category is TorchLensWarning and "torch.cos" in str(w.message)
        ]
        assert diagnostics, "epoch-2 foreign-patch skip stayed silent"
        assert "unwrapped" in str(diagnostics[0].message)
    finally:
        unwrap_torch()
        torch.cos = original_cos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrap_torch()


def test_failed_shim_install_purges_registry(monkeypatch) -> None:
    """sol 5 / fable 3: a failed install must not strand registry entries."""

    from torchlens.backends.torch import identity_shims as shims

    before_live = dict(shims._live_shims)
    if shims._family_installed:
        shims.remove_identity_shims()
    try:
        assert not shims._live_shims

        def _boom(records):  # noqa: ANN001, ANN202 - test shim
            raise RuntimeError("injected install failure")

        monkeypatch.setattr(shims, "_install_protocol_identity_shims", _boom)
        with pytest.raises(RuntimeError, match="injected install failure"):
            shims.install_identity_shims()
        assert not shims._live_shims, "failed install stranded shim objects in the live registry"
    finally:
        monkeypatch.undo()
        if not shims._family_installed and before_live:
            shims.install_identity_shims()
