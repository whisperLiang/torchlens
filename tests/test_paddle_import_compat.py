"""The lazy Paddle import guard changes only the affected native loader."""

from __future__ import annotations

import builtins
import ctypes
import importlib.util
import os
import sys
from collections.abc import Iterator
from importlib.machinery import ExtensionFileLoader, ModuleSpec, PathFinder, SourceFileLoader
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from torchlens import _paddle_compat as compat

pytestmark = pytest.mark.smoke

_GUARD_ID = "torchlens.paddle_llvm.v1"
_EXTENSION = "paddle.base.libpaddle"
_EXTENSION_PATH = "/mock/site-packages/paddle/base/libpaddle.so"
_CINN_PATH = "/mock/site-packages/paddle/libs/libcinnapi.so"


@pytest.fixture(autouse=True)
def _isolated_import_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate finder mutations and forbid interpreter-wide loader flag changes.

    Parameters
    ----------
    monkeypatch:
        Restore import machinery and platform capabilities after each check.

    Yields
    ------
    None
        A Linux-like import environment with no previously installed guard.
    """

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    compat.remove_import_guard()
    monkeypatch.setattr(sys, "platform", "linux")
    for name, value in (("RTLD_LOCAL", 0), ("RTLD_NOW", 2), ("RTLD_DEEPBIND", 8)):
        monkeypatch.setattr(os, name, value, raising=False)
    get_flags = getattr(sys, "getdlopenflags", lambda: None)
    original_dlopen_flags = get_flags()
    original_sys_flags = sys.flags
    set_flags = Mock(side_effect=AssertionError("global dlopen flags must not change"))
    monkeypatch.setattr(sys, "setdlopenflags", set_flags, raising=False)
    yield
    assert get_flags() == original_dlopen_flags
    assert sys.flags is original_sys_flags
    set_flags.assert_not_called()


def _extension_spec() -> tuple[ModuleSpec, ExtensionFileLoader]:
    """Return an ordinary extension spec without loading any native library.

    Returns
    -------
    tuple[ModuleSpec, ExtensionFileLoader]
        The spec and its original, unexecuted loader.
    """

    loader = ExtensionFileLoader(_EXTENSION, _EXTENSION_PATH)
    return ModuleSpec(_EXTENSION, loader, origin=_EXTENSION_PATH), loader


def test_install_is_idempotent_and_preserves_custom_finder_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert exactly one guard immediately before the standard path finder.

    Parameters
    ----------
    monkeypatch:
        Restore the synthetic finder list after the check.
    """

    before, after = object(), object()
    monkeypatch.setattr(sys, "meta_path", [before, PathFinder, after])
    assert compat.install_import_guard() is True
    installed = tuple(sys.meta_path)
    assert len(installed) == 4
    assert installed[0] is before and installed[2:] == (PathFinder, after)
    assert isinstance(installed[1], compat._PaddleImportGuard)
    assert installed[1].guard_id == _GUARD_ID
    assert compat.install_import_guard() is True
    assert tuple(sys.meta_path) == installed


def test_matching_standalone_guard_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A separately loaded copy is recognized by its shared marker, not its class.

    Parameters
    ----------
    monkeypatch:
        Restore the synthetic finder list after the check.
    """

    installed = SimpleNamespace(guard_id=_GUARD_ID)
    monkeypatch.setattr(sys, "meta_path", [installed])
    assert compat.install_import_guard() is True
    assert sys.meta_path == [installed]


@pytest.mark.parametrize("unsupported", ["platform", "deepbind", "path_finder"])
def test_unsupported_import_runtime_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, unsupported: str
) -> None:
    """Unsupported environments remain untouched rather than partially configured.

    Parameters
    ----------
    monkeypatch:
        Restore platform and import machinery after the check.
    unsupported:
        Capability missing from the simulated import environment.
    """

    if unsupported == "platform":
        monkeypatch.setattr(sys, "platform", "darwin")
    elif unsupported == "deepbind":
        monkeypatch.delattr(os, "RTLD_DEEPBIND")
    else:
        monkeypatch.setattr(sys, "meta_path", [object()])
    original = tuple(sys.meta_path)
    assert compat.install_import_guard() is False
    assert tuple(sys.meta_path) == original


def test_remove_only_removes_matching_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removal preserves unrelated finders, their order, and the list identity.

    Parameters
    ----------
    monkeypatch:
        Restore the synthetic finder list after the check.
    """

    unrelated = SimpleNamespace(guard_id="another.package.guard")
    standalone = SimpleNamespace(guard_id=_GUARD_ID)
    finders = [unrelated, compat._PaddleImportGuard(), PathFinder, standalone]
    monkeypatch.setattr(sys, "meta_path", finders)
    assert compat.remove_import_guard() is None
    assert sys.meta_path is finders
    assert finders == [unrelated, PathFinder]
    compat.remove_import_guard()
    assert finders == [unrelated, PathFinder]


def test_cold_module_import_and_install_do_not_import_frameworks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone helper and installation require only standard-library imports.

    Parameters
    ----------
    monkeypatch:
        Restore the import recorder after loading an independent helper copy.
    """

    original_import = builtins.__import__
    imports: list[str] = []

    def checked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Reject framework imports while recording ordinary module dependencies.

        Parameters
        ----------
        name:
            Module being imported.
        *args, **kwargs:
            Remaining arguments forwarded to Python's importer.

        Returns
        -------
        Any
            The ordinary import result.
        """

        imports.append(name)
        assert name.partition(".")[0] not in {"paddle", "tensorflow", "torch", "torchlens"}
        return original_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("_paddle_guard_unit_test", compat.__file__)
    assert spec is not None and spec.loader is not None
    standalone = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    spec.loader.exec_module(standalone)
    assert standalone.install_import_guard() is True
    assert imports
    assert all(name.partition(".")[0] in sys.stdlib_module_names for name in imports)


@pytest.mark.parametrize("fullname", ["paddle", "paddle.base", "tensorflow", "other.libpaddle"])
def test_unrelated_modules_never_consult_path_finder(
    monkeypatch: pytest.MonkeyPatch, fullname: str
) -> None:
    """Even with TensorFlow present, unrelated imports are never intercepted.

    Parameters
    ----------
    monkeypatch:
        Restore the module cache and mocked path finder after the check.
    fullname:
        Non-target module requested from the guard.
    """

    monkeypatch.setitem(sys.modules, "tensorflow", ModuleType("tensorflow"))
    lookup = Mock(side_effect=AssertionError("unrelated import consulted PathFinder"))
    monkeypatch.setattr(PathFinder, "find_spec", lookup)
    assert compat._PaddleImportGuard().find_spec(fullname) is None
    lookup.assert_not_called()


def test_paddle_first_import_is_not_intercepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a prior TensorFlow import, Paddle uses its ordinary import path.

    Parameters
    ----------
    monkeypatch:
        Restore the module cache and mocked path finder after the check.
    """

    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    lookup = Mock(side_effect=AssertionError("Paddle-first import consulted PathFinder"))
    monkeypatch.setattr(PathFinder, "find_spec", lookup)
    assert compat._PaddleImportGuard().find_spec(_EXTENSION) is None
    lookup.assert_not_called()


@pytest.mark.parametrize("unsupported", ["no_spec", "no_loader", "source_loader", "no_origin"])
def test_nonstandard_extension_is_not_wrapped(
    monkeypatch: pytest.MonkeyPatch, unsupported: str
) -> None:
    """Only a resolved native extension with a concrete origin can receive the guard.

    Parameters
    ----------
    monkeypatch:
        Restore mocked resolution, filesystem checks, and TensorFlow presence.
    unsupported:
        Unsupported result returned by the ordinary path finder.
    """

    spec, original = _extension_spec()
    if unsupported == "no_loader":
        spec.loader = None
    elif unsupported == "source_loader":
        spec.loader = SourceFileLoader(_EXTENSION, _EXTENSION_PATH)
    elif unsupported == "no_origin":
        spec.origin = None
    original_loader = spec.loader
    monkeypatch.setitem(sys.modules, "tensorflow", ModuleType("tensorflow"))
    lookup = Mock(return_value=None if unsupported == "no_spec" else spec)
    monkeypatch.setattr(PathFinder, "find_spec", lookup)
    file_check = Mock(side_effect=AssertionError("unsupported spec reached filesystem checks"))
    monkeypatch.setattr(os.path, "isfile", file_check)
    assert compat._PaddleImportGuard().find_spec(_EXTENSION) is None
    assert spec.loader is original_loader
    assert original.path == _EXTENSION_PATH
    file_check.assert_not_called()


def test_extension_without_bundled_cinn_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Builds without Paddle's adjacent CINN library retain their original loader.

    Parameters
    ----------
    monkeypatch:
        Restore mocked extension resolution and filesystem checks.
    """

    spec, original = _extension_spec()
    monkeypatch.setitem(sys.modules, "tensorflow", ModuleType("tensorflow"))
    monkeypatch.setattr(PathFinder, "find_spec", Mock(return_value=spec))
    file_check = Mock(return_value=False)
    monkeypatch.setattr(os.path, "isfile", file_check)
    assert compat._PaddleImportGuard().find_spec(_EXTENSION) is None
    file_check.assert_called_once_with(_CINN_PATH)
    assert spec.loader is original


def test_guard_preserves_spec_and_original_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarding preserves resolution metadata and forwards path/reload arguments.

    Parameters
    ----------
    monkeypatch:
        Restore mocked extension resolution and native-library loading.
    """

    spec, original = _extension_spec()
    state = object()
    spec.loader_state = state
    spec.cached = "/mock/original-cache"
    spec.has_location = True
    original_state = vars(spec).copy()
    search_path = ["/mock/site-packages/paddle/base"]
    target = ModuleType(_EXTENSION)
    monkeypatch.setitem(sys.modules, "tensorflow", ModuleType("tensorflow"))
    lookup = Mock(return_value=spec)
    monkeypatch.setattr(PathFinder, "find_spec", lookup)
    monkeypatch.setattr(os.path, "isfile", Mock(return_value=True))
    load_native = Mock(side_effect=AssertionError("finding a spec must not load native libraries"))
    monkeypatch.setattr(ctypes, "CDLL", load_native)

    assert compat._PaddleImportGuard().find_spec(_EXTENSION, search_path, target) is spec
    lookup.assert_called_once_with(_EXTENSION, search_path, target)
    assert isinstance(spec.loader, compat._PaddleExtensionLoader)
    assert spec.loader.original is original
    assert {key: value for key, value in vars(spec).items() if key != "loader"} == {
        key: value for key, value in original_state.items() if key != "loader"
    }
    load_native.assert_not_called()


def test_native_preload_precedes_create_and_loader_methods_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a local deep-bound handle before native initialization, then delegate.

    Parameters
    ----------
    monkeypatch:
        Restore native-library and original-loader mocks after the check.
    """

    spec, original = _extension_spec()
    module = ModuleType(_EXTENSION)
    handle = object()
    timeline = Mock()
    load_native = Mock(return_value=handle)
    create = Mock(return_value=module)
    execute = Mock()
    timeline.attach_mock(load_native, "preload")
    timeline.attach_mock(create, "create")
    timeline.attach_mock(execute, "execute")
    monkeypatch.setattr(ctypes, "CDLL", load_native)
    monkeypatch.setattr(original, "create_module", create)
    monkeypatch.setattr(original, "exec_module", execute)
    loader = compat._PaddleExtensionLoader(original)
    spec.loader = loader

    assert loader.create_module(spec) is module
    assert loader._handle is handle
    assert loader.exec_module(module) is None
    assert [call[0] for call in timeline.mock_calls] == ["preload", "create", "execute"]
    load_native.assert_called_once_with(
        _EXTENSION_PATH, mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_DEEPBIND
    )
    create.assert_called_once_with(spec)
    execute.assert_called_once_with(module)
    assert loader.path == original.path
    assert loader.get_filename(_EXTENSION) == original.get_filename(_EXTENSION)
    with pytest.raises(AttributeError):
        _ = loader.nonexistent_loader_attribute


@pytest.mark.parametrize("failure_stage", ["preload", "create", "execute"])
def test_loading_errors_propagate_without_an_unprotected_retry(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Native-loading and original-loader errors remain visible without fallback.

    Parameters
    ----------
    monkeypatch:
        Restore native-library and original-loader mocks after the check.
    failure_stage:
        Loader stage that raises the sentinel error.
    """

    spec, original = _extension_spec()
    module = ModuleType(_EXTENSION)
    error = OSError(f"sentinel {failure_stage} failure")
    load_native = Mock(return_value=object())
    create = Mock(return_value=module)
    execute = Mock()
    {"preload": load_native, "create": create, "execute": execute}[
        failure_stage
    ].side_effect = error
    monkeypatch.setattr(ctypes, "CDLL", load_native)
    monkeypatch.setattr(original, "create_module", create)
    monkeypatch.setattr(original, "exec_module", execute)
    loader = compat._PaddleExtensionLoader(original)
    spec.loader = loader
    with pytest.raises(OSError) as caught:
        created = loader.create_module(spec)
        loader.exec_module(created)
    assert caught.value is error
    assert load_native.call_count == 1
    assert create.call_count == (failure_stage != "preload")
    assert execute.call_count == (failure_stage == "execute")
