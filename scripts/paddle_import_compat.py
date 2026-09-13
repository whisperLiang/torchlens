"""Install/check/remove the opt-in Paddle LLVM import guard in the active venv.

Only two TorchLens-owned startup files are managed. Framework wheels, dependency
versions, the system interpreter, and the import search path are not modified.
"""

from __future__ import annotations

import argparse
import os
import sys
import sysconfig
import tempfile
from pathlib import Path

_HEADER = "# TorchLens-managed Paddle LLVM import compatibility v1\n"
_MODULE_NAME = "_torchlens_paddle_compat.py"
_PTH_NAME = "_torchlens_paddle_compat.pth"
_PTH_CONTENT = (
    _HEADER + "import _torchlens_paddle_compat; _torchlens_paddle_compat.install_import_guard()\n"
)


def _site_packages() -> Path:
    """Resolve the active virtual environment's site-packages without following overrides.

    Returns
    -------
    Path
        Existing site-packages directory within the active virtual environment.
    """

    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use a virtual environment's Python; system installation is refused.")
    prefix = Path(sys.prefix).resolve()
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    if not site_packages.is_relative_to(prefix) or not site_packages.is_dir():
        raise RuntimeError("The active virtual environment has no local site-packages directory.")
    return site_packages


def _managed_paths(site_packages: Path) -> tuple[Path, Path]:
    """Validate exact managed targets before making any installation changes.

    Parameters
    ----------
    site_packages:
        Validated virtual-environment directory.

    Returns
    -------
    tuple of Path
        Module and startup-file paths; existing unowned files cause refusal.
    """

    paths = (site_packages / _MODULE_NAME, site_packages / _PTH_NAME)
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"Refusing a symlink at managed target: {path}")
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"Refusing a non-file at managed target: {path}")
            with path.open(encoding="utf-8") as existing:
                if existing.readline() != _HEADER:
                    raise RuntimeError(
                        f"Refusing to overwrite a file not owned by this tool: {path}"
                    )
    return paths


def _atomic_write(path: Path, content: str) -> None:
    """Publish one owned startup file atomically.

    Parameters
    ----------
    path:
        Exact target already validated by ``_managed_paths``.
    content:
        Complete managed contents.
    """

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Manage the guard for the interpreter running this script.

    Parameters
    ----------
    argv:
        Optional CLI argument list for tests and programmatic invocation.

    Returns
    -------
    int
        Zero on success, one when a check finds an absent or stale installation.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "check", "remove"))
    args = parser.parse_args(argv)
    module_path, pth_path = _managed_paths(_site_packages())
    if args.action == "remove":
        # Stop future startup imports before removing their implementation.
        pth_path.unlink(missing_ok=True)
        module_path.unlink(missing_ok=True)
        print("Removed the two managed startup files; restart Python to apply.")
        return 0

    source = Path(__file__).resolve().parents[1] / "torchlens" / "_paddle_compat.py"
    module_content = _HEADER + source.read_text(encoding="utf-8")
    expected = ((module_path, module_content), (pth_path, _PTH_CONTENT))
    if args.action == "check":
        current = all(
            path.is_file() and path.read_text(encoding="utf-8") == text for path, text in expected
        )
        print(
            "Paddle import guard is current."
            if current
            else "Paddle import guard is absent or stale."
        )
        return 0 if current else 1

    if sys.platform != "linux" or not hasattr(os, "RTLD_DEEPBIND"):
        raise RuntimeError("This compatibility guard requires Linux with RTLD_DEEPBIND.")
    for path, content in expected:
        _atomic_write(path, content)
    print(
        f"Installed Paddle import guard in {module_path.parent}; applies to new Python processes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
