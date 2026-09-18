"""HF text defaults honor both explicit flat and grouped output-style choices."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import torchlens as tl
from torchlens._deprecations import MISSING
from torchlens.bridge import hf

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("style", [None, "classification", "hf_text"])
@pytest.mark.parametrize("grouped", [False, True])
def test_explicit_output_style_is_not_overridden(
    style: str | None, grouped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even explicit None remains an opt-out of bridge-provided decoding."""

    forwarded: dict[str, Any] = {}

    def capture(model: Any, text: Any, **kwargs: Any) -> Any:
        """Inspect forwarded options without requiring Transformers or execution."""

        forwarded.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(tl, "trace", capture)
    tokenizer = SimpleNamespace(name_or_path="local")
    options = (
        {"capture": tl.options.CaptureOptions(output_style=style), "output_style": MISSING}
        if grouped
        else {"output_style": style}
    )
    hf.trace_text(SimpleNamespace(), "hello", tokenizer=tokenizer, **options)
    if grouped:
        assert forwarded["capture"].output_style == style
        assert forwarded["output_style"] is MISSING
    else:
        assert forwarded["output_style"] == style
