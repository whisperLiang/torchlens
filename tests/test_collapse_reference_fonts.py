"""Reference-only font pinning must never leak into ordinary user rendering."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("previous", (None, "/tmp/custom-fontconfig.conf"))
@pytest.mark.parametrize("raises", (False, True))
def test_reference_render_restores_fontconfig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, previous: str | None, raises: bool
) -> None:
    """Both successful and failed renders leave the exact caller setting intact."""

    from scripts import render_collapse_reference as gallery

    if previous is None:
        monkeypatch.delenv("FONTCONFIG_FILE", raising=False)
    else:
        monkeypatch.setenv("FONTCONFIG_FILE", previous)
    observed: list[Path] = []

    def render_probe(out_dir: Path, *, copy_for_review: bool) -> None:
        """Observe the scoped config at the renderer's entry point."""

        assert out_dir == tmp_path
        assert copy_for_review is False
        config = Path(os.environ["FONTCONFIG_FILE"])
        assert config == Path(gallery.__file__).with_name("collapse_reference_fonts.conf")
        assert "DejaVu Serif" in config.read_text(encoding="utf-8")
        observed.append(config)
        if raises:
            raise RuntimeError("render failed")

    monkeypatch.setattr(gallery, "_render_gallery", render_probe)
    if raises:
        with pytest.raises(RuntimeError, match="render failed"):
            gallery.render(tmp_path, copy_for_review=False)
    else:
        gallery.render(tmp_path, copy_for_review=False)
    assert len(observed) == 1
    assert os.environ.get("FONTCONFIG_FILE") == previous
