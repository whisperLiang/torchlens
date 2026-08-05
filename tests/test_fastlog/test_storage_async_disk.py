"""Async disk-mode placeholder tests for fastlog v1."""

from __future__ import annotations

import pytest


@pytest.mark.rare
@pytest.mark.skip(reason="placeholder: async disk storage is intentionally not shipped in v1")
def test_async_disk_storage_pending() -> None:
    """Placeholder for a future async-disk test once the feature exists."""
