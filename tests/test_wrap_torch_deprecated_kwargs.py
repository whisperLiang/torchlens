"""r-b4 R48 (wrappers half): every explicit deprecated wrap_torch kwarg warns.

The truthiness guard (`if patch_policy is not None or patch_modules:`) accepted
and silently swallowed `patch_modules=[]`/`()`/`{}` and an explicit
`patch_policy=None` -- the exact silent-deprecation shape the census scanner
cannot see. MISSING sentinel defaults make ANY explicit pass warn.
"""

from __future__ import annotations

import warnings

import pytest

from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("empty", [(), [], {}])
def test_empty_patch_modules_warns(empty: object) -> None:
    """An explicit empty container is an explicit use of the deprecated kwarg."""

    try:
        with pytest.warns(DeprecationWarning, match="patch_modules"):
            wrap_torch(patch_modules=empty)  # type: ignore[arg-type]
    finally:
        unwrap_torch()


def test_explicit_none_patch_policy_warns() -> None:
    """An explicit patch_policy=None is an explicit use of the deprecated kwarg."""

    try:
        with pytest.warns(DeprecationWarning, match="patch_policy"):
            wrap_torch(patch_policy=None)
    finally:
        unwrap_torch()


def test_truthy_patch_policy_still_warns() -> None:
    """The historical truthy-value warning is unchanged."""

    try:
        with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
            wrap_torch(patch_policy="all")
    finally:
        unwrap_torch()


def test_omitted_kwargs_do_not_warn() -> None:
    """Plain wrap_torch() stays warning-free."""

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrap_torch()
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []
    finally:
        unwrap_torch()
