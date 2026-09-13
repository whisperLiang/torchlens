"""The Paddle compatibility installer owns only two files in the active venv."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import paddle_import_compat as installer

pytestmark = pytest.mark.smoke


@pytest.fixture
def installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route installation to a private directory without touching the active venv.

    Parameters
    ----------
    tmp_path:
        Per-test scratch directory.
    monkeypatch:
        Restore the simulated platform and installation path after the test.
    """

    monkeypatch.setattr(installer, "_site_packages", lambda: tmp_path)
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.os, "RTLD_DEEPBIND", 8, raising=False)
    return tmp_path


def test_install_check_update_and_remove_only_owned_files(installation: Path) -> None:
    """A reversible installation neither rewrites nor removes neighboring files.

    Parameters
    ----------
    installation:
        Isolated site-packages stand-in.
    """

    neighbor = installation / "unrelated.pth"
    neighbor.write_text("# user-owned startup file\n", encoding="utf-8")
    assert installer.main(["check"]) == 1
    assert installer.main(["install"]) == 0
    assert installer.main(["check"]) == 0
    module = installation / installer._MODULE_NAME
    startup = installation / installer._PTH_NAME
    assert module.read_text(encoding="utf-8").startswith(installer._HEADER)
    assert startup.read_text(encoding="utf-8") == installer._PTH_CONTENT
    # The copy stays standalone: installation never adds the checkout to sys.path.
    assert "sys.path" not in startup.read_text(encoding="utf-8")
    module.write_text(installer._HEADER + "# stale owned copy\n", encoding="utf-8")
    assert installer.main(["check"]) == 1
    assert installer.main(["install"]) == 0
    assert installer.main(["check"]) == 0
    assert installer.main(["remove"]) == 0
    assert not module.exists() and not startup.exists()
    assert neighbor.read_text(encoding="utf-8") == "# user-owned startup file\n"
    assert installer.main(["remove"]) == 0


@pytest.mark.parametrize("name", [installer._MODULE_NAME, installer._PTH_NAME])
@pytest.mark.parametrize("action", ["install", "remove"])
def test_unknown_target_is_never_overwritten_or_removed(
    installation: Path, name: str, action: str
) -> None:
    """Validate both targets before modifying either one.

    Parameters
    ----------
    installation:
        Isolated site-packages stand-in.
    name:
        One of the two installer target names.
    action:
        Mutating command whose preflight must refuse an unowned file.
    """

    target = installation / name
    target.write_text("# not owned by TorchLens\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in installation.iterdir()}
    with pytest.raises(RuntimeError, match="not owned"):
        installer.main([action])
    assert {path.name: path.read_bytes() for path in installation.iterdir()} == before


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_non_regular_managed_target_refuses(installation: Path, tmp_path: Path, kind: str) -> None:
    """Directories and symlinks cannot redirect the two-file installer.

    Parameters
    ----------
    installation:
        Isolated site-packages stand-in.
    tmp_path:
        Scratch directory containing a separate symlink destination.
    kind:
        Non-regular target form to refuse.
    """

    target = installation / installer._PTH_NAME
    if kind == "directory":
        target.mkdir()
    else:
        destination = tmp_path / "user-file"
        destination.write_text(installer._HEADER, encoding="utf-8")
        try:
            target.symlink_to(destination)
        except OSError:
            pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(RuntimeError, match="Refusing"):
        installer.main(["install"])
    assert not (installation / installer._MODULE_NAME).exists()


def test_system_interpreter_installation_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never install an interpreter-wide hook outside a virtual environment.

    Parameters
    ----------
    monkeypatch:
        Restore the active interpreter prefix after simulating system Python.
    """

    monkeypatch.setattr(installer.sys, "prefix", installer.sys.base_prefix)
    with pytest.raises(RuntimeError, match="system installation is refused"):
        installer._site_packages()


def test_site_packages_must_remain_inside_active_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a sysconfig path outside the interpreter's own environment.

    Parameters
    ----------
    tmp_path:
        Scratch directory containing separate venv and foreign directories.
    monkeypatch:
        Restore the simulated prefix and sysconfig lookup.
    """

    prefix = tmp_path / "venv"
    foreign = tmp_path / "foreign"
    prefix.mkdir()
    foreign.mkdir()
    monkeypatch.setattr(installer.sys, "prefix", str(prefix))
    monkeypatch.setattr(installer.sysconfig, "get_path", lambda name: str(foreign))
    with pytest.raises(RuntimeError, match="no local site-packages"):
        installer._site_packages()


def test_install_on_unsupported_platform_leaves_no_files(
    installation: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The platform gate runs before publishing either startup file.

    Parameters
    ----------
    installation:
        Isolated site-packages stand-in.
    monkeypatch:
        Restore the simulated unsupported platform after the test.
    """

    monkeypatch.setattr(installer.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        installer.main(["install"])
    assert list(installation.iterdir()) == []
