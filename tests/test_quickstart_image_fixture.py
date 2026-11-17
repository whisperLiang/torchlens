"""The README image is fixed test input, independent of network and user caches."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from tests import test_quickstart_readme_gate as gate


def test_readme_image_needs_no_network_or_user_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read the exact README image with an empty cache and network access refused."""

    def refuse_network(*args: object, **kwargs: object) -> None:
        """Fail if fixture loading tries to contact a remote server."""
        raise AssertionError("README image fixture attempted a network request")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", refuse_network)
    data = gate._pinned_screen_image()
    assert data == (gate._REPO_ROOT / "tests/fixtures/quickstart/cat.jpg").read_bytes()


@pytest.mark.parametrize("corruption", ["truncated", "same_length"])
def test_readme_image_corruption_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str
) -> None:
    """Neither a truncated fixture nor different bytes with the same length are accepted."""
    data = gate._pinned_screen_image()
    corrupted = data[:-1] if corruption == "truncated" else bytes([data[0] ^ 1]) + data[1:]
    path = tmp_path / "cat.jpg"
    path.write_bytes(corrupted)
    monkeypatch.setattr(gate, "_SCREEN_IMAGE_PATH", path)
    with pytest.raises(AssertionError, match="not the pinned README image|not the pinned image"):
        gate._pinned_screen_image()
