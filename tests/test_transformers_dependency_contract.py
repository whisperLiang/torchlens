"""HF and test extras must admit the same Transformers major versions."""

from __future__ import annotations

from importlib.metadata import requires

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.smoke


def test_hf_and_test_extras_share_transformers_4_and_5_support() -> None:
    """The installed package metadata cannot reintroduce the RFDETR/test conflict."""

    requirements = [Requirement(row) for row in requires("torchlens") or ()]
    by_extra = {
        extra: [
            requirement.specifier
            for requirement in requirements
            if requirement.name == "transformers"
            and requirement.marker is not None
            and requirement.marker.evaluate({"extra": extra})
        ]
        for extra in ("hf", "test")
    }
    assert len(by_extra["hf"]) == len(by_extra["test"]) == 1
    assert by_extra["hf"][0] == by_extra["test"][0]
    supported = by_extra["hf"][0]
    for version in ("4.45.0", "4.57.6", "5.1.0", "5.17.0"):
        assert supported.contains(version), version
    for version in ("4.44.2", "6.0.0"):
        assert not supported.contains(version), version
