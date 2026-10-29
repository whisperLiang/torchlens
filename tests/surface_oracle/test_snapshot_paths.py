"""Source paths have one portable spelling in public-surface snapshots."""

import pytest

from . import _snapshot

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_repository_source_locations_use_portable_separators(
    monkeypatch: pytest.MonkeyPatch, separator: str
) -> None:
    """Native file paths retain their relative path and source line."""
    root = separator.join(("C:", "work", "torchlens"))
    monkeypatch.setattr(_snapshot, "_REPO_ROOT", root)
    source = separator.join((root, "tests", "model.py")) + ":11"
    assert _snapshot.canonicalize(source) == "<repo>/tests/model.py:11"


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_installed_source_locations_use_portable_separators(separator: str) -> None:
    """Dependency paths hide the environment prefix on either platform."""
    source = separator.join(("C:", "venv", "site-packages", "torch", "nn", "module.py"))
    assert _snapshot.canonicalize(source) == "<site>/torch/nn/module.py"


def test_non_path_backslashes_are_preserved() -> None:
    """Canonicalization must preserve escape sequences in ordinary public strings."""
    text = r"literal\ntext\twith\backslashes"
    assert _snapshot.canonicalize(text) == text
