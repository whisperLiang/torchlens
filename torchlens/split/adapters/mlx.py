"""MLX split adapter shell."""

from __future__ import annotations

from ._unsupported import UnsupportedSplitAdapter


class MlxSplitAdapter(UnsupportedSplitAdapter):
    """MLX split capability shell for the deferred implementation."""

    def __init__(self) -> None:
        """Create the MLX adapter shell."""

        super().__init__("mlx")


__all__ = ["MlxSplitAdapter"]
