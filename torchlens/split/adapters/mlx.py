"""MLX split adapter shell."""

from __future__ import annotations

from ._unsupported import UnsupportedSplitAdapter


class MlxSplitAdapter(UnsupportedSplitAdapter):
    """MLX split capabilities for v1."""

    def __init__(self) -> None:
        """Create the MLX adapter shell."""

        super().__init__("mlx")


__all__ = ["MlxSplitAdapter"]
